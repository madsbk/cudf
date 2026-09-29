# cudf-polars + PyTorch Distributed

This document describes how `SPMDEngine` runs inside a `torch.distributed`
job, so that a single `torchrun` job can do distributed GPU ETL with Polars
and feed the result into a multi-GPU PyTorch training loop.

> **Status:** experimental. The API lives under `cudf_polars.engine` and
> may change without notice.

It assumes familiarity with the SPMD engine itself. The
[SPMD engine guide](../../../docs/cudf/source/cudf_polars/spmd_engine.md)
covers how ranks are launched, how file-based scans are partitioned across
ranks, the query symmetry requirement, and how rank-local results are
collected. This document covers only what is specific to running that
engine inside a `torch.distributed` job.

## Contents

* [Why this exists](#why-this-exists)
* [Mental model](#mental-model)
* [Quickstart](#quickstart)
* [Reading input](#reading-input)
* [Operational guidance](#operational-guidance)
* [Pitfalls](#pitfalls)

---

## Why this exists

`SPMDEngine` and `torch.distributed` are both SPMD runtimes. Every rank
runs the same script, owns a rank-local slice of data, and coordinates
with its peers through collective operations. Running both in one process
per rank makes several workflows simple that otherwise need separate jobs
or clusters.

The snippets below run inside the `with` block from the
[Quickstart](#quickstart), and `events`, `users` and `catalog` stand for
`pl.scan_parquet(...)` inputs.

### Collapse two-cluster ETL-to-training pipelines into one

A common production layout today is:

1. A CPU cluster (Spark or Dask) builds training tables from raw events.
2. The tables are written to object storage as Parquet.
3. A GPU cluster loads the Parquet files and trains.

With cudf-polars and PyTorch in one process group, a single `torchrun`
job does the ETL on the GPUs that will train the model. The CPU cluster,
the intermediate Parquet write and the orchestration between the two jobs
all go away, and the GPUs no longer sit idle while the ETL runs.

```python
features = (
    events.group_by("user_id")
    .agg(
        pl.col("amount").sum().alias("spend"),
        pl.len().alias("n_events"),
        pl.col("label").max(),
    )
    .select(pl.col("spend", "n_events", "label").cast(pl.Float32))
)
tensors = persisted_to_torch(engine.execute(features), engine=engine, ensure_sharded=True)
```

### Per-epoch reshuffle and feature recompute

Tabular deep learning (recsys, ads, fraud detection) often wants a fresh
global shuffle each epoch, and sometimes wants to recompute features such
as target encodings between epochs. When that round-trips through a CPU
cluster, trainers tend to cache one shuffle for the whole run. On the
training GPUs it is cheap enough to do every epoch.

```python
for epoch in range(num_epochs):
    # Target encoding, recomputed from the current events.
    merchant_rate = events.group_by("merchant_id").agg(
        pl.col("label").mean().alias("merchant_rate")
    )
    epoch_query = (
        events.join(merchant_rate, on="merchant_id")
        # A fresh global shuffle: sort on a hash seeded by the epoch.
        .sort(pl.col("event_id").hash(seed=epoch))
        .select(pl.col("amount", "merchant_rate", "label").cast(pl.Float32))
    )
    batch = persisted_to_torch(engine.execute(epoch_query), engine=engine, ensure_sharded=True)
    train_one_epoch(model, batch)
```

### Big distributed joins for training data prep

Building a training table for a deep recommender often means joining
users, events and catalog, which does not fit on one GPU. cudf-polars
runs the join across the ranks, and the result lands on the same ranks as
the DDP or FSDP model without being written to disk.

```python
train_table = (
    events.join(users, on="user_id")
    .join(catalog, on="item_id")
    .select(pl.col("age", "price", "label").cast(pl.Float32))
)
tensors = persisted_to_torch(engine.execute(train_table), engine=engine, ensure_sharded=True)
```

### Active learning

An active learning loop trains, scores the whole corpus, picks the
uncertain rows, relabels them and retrains. Scoring and picking is
distributed ETL and retraining is DDP, so the whole loop becomes a plain
Python `for` loop in one script instead of a multi-job pipeline.

```python
for _ in range(num_rounds):
    train(model, labeled_tensors())
    # Score this rank's share of the unlabeled pool.
    pool = persisted_to_torch(
        engine.execute(unlabeled.select(pl.col("id"), pl.col("feat").cast(pl.Float32))),
        engine=engine,
        ensure_sharded=True,
    )
    with torch.no_grad():
        p = model(pool["feat"].unsqueeze(1)).sigmoid().squeeze(1)
    # Pick the globally most uncertain rows. Every rank gets the same answer.
    scores = pl.LazyFrame({
        "id": pool["id"].cpu().numpy(),
        "margin": (p - 0.5).abs().cpu().numpy(),
    })
    picked = scores.sort("margin").head(1000).collect(engine=engine)
    if dist.get_rank() == 0:
        send_for_labeling(picked["id"])
```

### Last-mile transforms

Transforms right before the forward pass, such as normalization or
frequency capping, can run in the query. The result becomes torch tensors
on the same GPU without a round trip through Arrow or Parquet.

```python
query = events.select(
    # Normalize with the global mean and standard deviation.
    ((pl.col("amount") - pl.col("amount").mean()) / pl.col("amount").std()).cast(pl.Float32),
    # Frequency cap.
    pl.col("clicks").clip(upper_bound=100).cast(pl.Float32),
    pl.col("label").cast(pl.Float32),
)
tensors = persisted_to_torch(engine.execute(query), engine=engine, ensure_sharded=True)
```

### Tabular features next to torch models

Tabular features from cudf-polars and embedding lookups or image models
in torch can serve the same batch on the same ranks, in one script
instead of two services.

```python
rows = persisted_to_torch(
    engine.execute(
        events.select(pl.col("item_id").cast(pl.Int64), pl.col("amount", "label").cast(pl.Float32))
    ),
    engine=engine,
    ensure_sharded=True,
)
item_embedding = torch.nn.Embedding(num_items, 64).cuda()
x = torch.cat([item_embedding(rows["item_id"]), rows["amount"].unsqueeze(1)], dim=1)
```

### When you do *not* need this

If your ETL fits on a single GPU, or runs once a day and is already
cached, this is overkill. Use a single-process cudf-polars job or
`cudf.pandas` instead. The integration pays off when the ETL is both large
enough to need several GPUs and close enough to training to benefit from
sharing them.

---

## Mental model

Both runtimes are SPMD:

| Concept              | cudf-polars (`SPMDEngine`) | PyTorch (`torch.distributed`) |
| -------------------- | -------------------------- | ----------------------------- |
| Identity             | `engine.rank`              | `dist.get_rank()`             |
| World size           | `engine.nranks`            | `dist.get_world_size()`       |
| Device               | CUDA device 0              | `cuda:0`, after `use_gpu()`   |
| Collective transport | UCXX                       | NCCL (typical)                |

Two things must agree across the runtimes:

1. **World size.** `from_torch_distributed()` takes it from the torch
   process group, so it always matches. Rank numbers may differ, see
   [Rank numbering](#rank-numbering).
2. **Device.** Both runtimes must use the same GPU on each rank.
   `use_gpu()` arranges this before any CUDA call, as in the Quickstart.

The two transports are independent. UCXX moves DataFrame fragments during
ETL and NCCL moves gradients during training. They share GPUs but not
communication channels.

---

## Quickstart

A complete script:

```python
# Launch: torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) script.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import polars as pl

from cudf_polars.engine.spmd import SPMDEngine, use_gpu
from cudf_polars.engine.torch_interop import persisted_to_torch

# Give this rank its GPU before any CUDA call, including NCCL init.
use_gpu(int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(0)
dist.init_process_group(backend="nccl")

# 1) Distributed GPU ETL. Every rank passes the same paths and the engine
#    splits the scan between them. The result stays on the GPU.
with SPMDEngine.from_torch_distributed() as engine:
    result = engine.execute(
        pl.scan_parquet("s3://bucket/data/*.parquet")
        .filter(pl.col("label").is_not_null())
        .select(pl.col("feat").cast(pl.Float32), pl.col("label").cast(pl.Float32))
    )

    # 2) Hand off to torch as a zero-copy view of that GPU memory. Clone so
    #    the tensors outlive the engine.
    tensors = persisted_to_torch(result, engine=engine, ensure_sharded=True)
    feat, label = tensors["feat"].clone(), tensors["label"].clone()
    del tensors

# 3) Train normally with DDP.
model = DDP(MyModel().cuda(), device_ids=[0])
train(model, feat, label)

dist.destroy_process_group()
```

What each step does:

1. `from_torch_distributed()` uses the torch process group to bootstrap
   the engine's UCXX communicator. It takes an optional
   `StreamingOptions`, read the same way as in `SPMDEngine.from_options`.
   Leaving the `with` block shuts the engine down.
2. `engine.execute(lf)` keeps each rank's result on its GPU.
   `persisted_to_torch` views each column as a tensor sharing that GPU
   memory. Columns must be numeric or boolean and contain no nulls. It
   consumes the result, so the result cannot also be collected.
3. The ETL ranks and the DDP ranks are the same processes, so the
   per-rank tensors are what DDP expects.

A result is either sharded, each rank holding different rows, or
replicated, every rank holding the same full copy. Which one a query
produces depends on how the engine partitioned it, so it cannot be read
off the query. Replicated is right for values every rank needs, such as
normalization constants, so leave `ensure_sharded` off for those. For
training data, `ensure_sharded=True` cuts a replicated result down to
this rank's share, split the way `torch.chunk` would split it, and
returns a sharded result unchanged.

---

## Reading input

Nothing torch-specific applies here. File-based sources (`scan_parquet`,
`scan_csv`, ...) are split automatically so that each rank reads a
different file or row-group range, and in-memory frames are already
rank-local. Pass every rank the same paths and let the engine split them.
Do not hand each rank its own file list.

A dataset small enough to fit in a single partition is read by a single
rank, so a toy dataset does not exercise multi-rank IO even though the
results are correct.

---

## Operational guidance

### Launching

Launch with `torchrun`:

```
torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) script.py
```

`from_torch_distributed()` takes rank and world size from the torch
process group. Do not combine it with `rrun`. A job launched with `rrun`
should use plain `SPMDEngine()` instead.

### Device placement

This is where the integration departs from the usual `torchrun` idiom,
which leaves every GPU visible and selects one with
`torch.cuda.set_device(LOCAL_RANK)`. The engine runs on CUDA device
ordinal 0, so its GPU has to come first in `CUDA_VISIBLE_DEVICES`.
`use_gpu` does that:

```python
use_gpu(int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(0)
```

Call it before any CUDA call, including
`dist.init_process_group(backend="nccl")`. Afterwards the process sees
exactly one GPU, numbered 0, so use `device="cuda:0"` for tensors and
`device_ids=[0]` for `DistributedDataParallel`. Constructing an engine on
a rank whose current device is not ordinal 0 raises. The
[SPMD engine guide](../../../docs/cudf/source/cudf_polars/spmd_engine.md)
explains the requirement.

### Backend choice

Use `backend="nccl"` for the torch process group, as DDP expects. The
engine uses the torch group only to exchange the UCXX address at startup.
Its own collectives go through UCXX.

### Optional: share the RMM pool with torch

By default, cuDF and torch each manage their own GPU memory. For
workloads that stress the allocator, such as large joins next to large
model states, torch can allocate from RMM too:

```python
import rmm.allocators.torch
torch.cuda.memory.change_current_allocator(
    rmm.allocators.torch.rmm_torch_allocator
)
```

This changes torch's global allocator, so it is opt-in.

### Shutdown order

Tear down in the reverse order of setup:

```python
with SPMDEngine.from_torch_distributed() as engine:
    ...  # ETL and handoff

# ... training ...

dist.destroy_process_group()
```

Leaving the `with` block releases the UCXX communicator, the RapidsMPF
context and the engine's threads. `dist.destroy_process_group()` then
releases the torch process group and NCCL communicators.

---

## Pitfalls

### Query symmetry

All ranks must issue the same Polars queries in the same order.
Collective operations are matched across ranks by an op id, so a
rank-conditional `collect`, an early exit, or any branch that makes ranks
run different queries deadlocks. This is the same rule as for plain
`SPMDEngine` use.

### Tensors pin the engine's memory pool

`persisted_to_torch` returns zero-copy views into the engine's memory
pool, which cannot be released while they are alive. `.clone()` the
tensors you want to keep and drop the views.

### Rank numbering

`engine.rank` is the UCXX communicator's numbering, assigned in the order
ranks connect, and it need not equal `dist.get_rank()`. The same holds
for `RRUN_RANK` under `rrun`.

Each rank stores and reads its own data under its own engine rank, and
gradient all-reduce does not care which rank holds which rows, so this is
usually harmless. It breaks code that treats the two numberings as one,
such as choosing input files by `dist.get_rank()` and assuming the engine
labels that data the same way. Pick one numbering per purpose.
