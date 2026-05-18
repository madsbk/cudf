# cudf-polars + PyTorch Distributed

This document describes how the `SPMDEngine` interoperates with PyTorch's
`torch.distributed` runtime so that a single `torchrun` job can do
distributed GPU ETL with Polars and then feed the result into a
multi-GPU PyTorch training loop.

> **Status:** experimental. The API lives under
> `cudf_polars.engine` and may change without notice.

This assumes familiarity with the SPMD engine itself. See the
[SPMD engine guide](../../../docs/cudf/source/cudf_polars/spmd_engine.md)
for how ranks are launched, how file-based scans are partitioned across
ranks, the query symmetry requirement, and how rank-local results are
collected. This document covers only what is specific to running that
engine inside a `torch.distributed` job.

## Contents

* [Why this exists](#why-this-exists)
* [Mental model](#mental-model)
* [Quickstart](#quickstart)
* [Reading input](#reading-input)
* [API reference](#api-reference)
* [Operational guidance](#operational-guidance)
* [Pitfalls](#pitfalls)
* [Future work: DTensor](#future-work-dtensor)
* [Sketch: TensorFlow](#sketch-applying-the-same-pattern-to-tensorflow)

---

## Why this exists

`SPMDEngine` and `torch.distributed` are both SPMD/MPI-style runtimes:
every rank runs the same script, owns a rank-local slice of data, and
coordinates with peers through collective operations. They are a natural
fit for each other, and combining them in one process per rank unlocks a
set of workflows that are otherwise awkward or expensive to express.

### Collapse two-cluster ETL-to-training pipelines into one

A common production layout today is:

1. CPU cluster (Spark or Dask) builds training tables from raw events.
2. Tables are written to object storage as Parquet.
3. GPU cluster loads the Parquet files and trains.

With cudf-polars + PyTorch in one process group, the CPU cluster
disappears. A single `torchrun` job does the ETL on the same GPUs that
will run the model. The intermediate Parquet write, the cross-cluster
handoff, and the Airflow stitching between the two clusters all go
away.

### Per-epoch reshuffle and on-the-fly feature recompute

Tabular deep learning (recsys, ads, fraud detection) usually wants a
fresh global shuffle each epoch, and sometimes wants to recompute
windowed features (rolling means, target encodings, frequency caps)
between epochs. When the shuffle has to round-trip through a CPU
cluster, the cost dominates and trainers cache shuffles for an entire
run. Running the shuffle on the training GPUs makes it cheap enough to
do inline, so each epoch sees genuinely fresh data.

### Big distributed joins for training data prep

Building a training table for a deep recommender often means joining
users x events x catalog, which does not fit on one GPU. cudf-polars
performs the distributed join on the SPMD ranks, and the result lands on
the same ranks as the DDP/FSDP model. The joined table never needs to
be materialized to disk.

### Active learning and human-in-the-loop loops

A typical active learning loop is: train, score the whole corpus, pick
uncertain rows, relabel, retrain. The "score and filter" half is
distributed GPU ETL; the "retrain" half is DDP. With this integration
both halves live in one script with one process group, so the loop
becomes a straight Python `for` loop instead of a multi-job DAG.

### Keep GPUs hot during data prep

In a split pipeline the GPU fleet sits idle while the CPU cluster runs.
Folding ETL into the training job uses the GPUs the user is already
paying for, which is the most direct way to improve utilization on
multi-GPU boxes.

### Zero-copy handoff for last-mile transforms

Last-mile transforms (target encoding, frequency capping, session
bucketing, normalization) right before the forward pass do not need to
round-trip through Arrow or Parquet. cuDF columns become torch tensors
on the same device through `__cuda_array_interface__`, sharing the same
memory, so the "final mile" of preprocessing stays on the GPU.

### Multi-modal pipelines in one process group

Tabular features (cudf-polars), image decode, and embedding lookups
(torch) for the same batch can live in the same world, the same ranks,
and the same NCCL group. A join key on the tabular side and an
`all_gather` of embeddings on the torch side are in the same script,
not in two services.

### When you do *not* need this

If your ETL fits on a single GPU, or runs once a day and is already
cached, this integration is overkill. Reach for a single-process
cudf-polars job, or `cudf.pandas`, instead. The integration earns its
keep when ETL is *both* large enough to need multi-GPU *and* close
enough to training to benefit from sharing the process group.

---

## Mental model

Both runtimes are SPMD:

| Concept            | cudf-polars (`SPMDEngine`) | PyTorch (`torch.distributed`) |
| ------------------ | -------------------------- | ----------------------------- |
| Identity           | `engine.rank`              | `dist.get_rank()`             |
| World size         | `engine.comm.nranks`       | `dist.get_world_size()`       |
| Device per rank    | current CUDA device        | `LOCAL_RANK` -> `set_device`  |
| Collective transport | UCXX                     | NCCL (typical)                |

Two invariants must hold across both runtimes:

1. **Same world size.** `engine.comm.nranks == dist.get_world_size()`,
   which `from_torch_distributed()` guarantees by reading the world size
   off the torch group. The *rank numbers* are a different matter, see
   "Rank numbering" under Pitfalls.
2. **Device alignment.** The current CUDA device on each rank is the
   same for both runtimes. `use_gpu()` arranges this before any CUDA
   call, as shown in the quickstart.

The two transports (UCXX and NCCL) operate independently. UCXX moves
DataFrame fragments during ETL; NCCL moves gradients during training.
They share GPUs but not communication channels.

---

## Quickstart

A complete `torchrun`-launched script:

```python
# script.py
# Launch:
#   torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) script.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import polars as pl

from cudf_polars.engine.spmd import SPMDEngine, use_gpu
from cudf_polars.engine.torch_interop import persisted_to_torch

# Give this rank its GPU *before* any CUDA work happens, including NCCL init.
use_gpu(int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(0)
dist.init_process_group(backend="nccl")

# 1) Distributed GPU ETL. Every rank passes the same paths; the engine
#    partitions the scan so each rank reads a different slice.
#    `execute` keeps each rank's result on the GPU.
with SPMDEngine.from_torch_distributed() as engine:
    result = engine.execute(
        pl.scan_parquet("s3://bucket/data/*.parquet")
        .filter(pl.col("label").is_not_null())
        .select(pl.col("feat").cast(pl.Float32), pl.col("label").cast(pl.Float32))
    )

    # 2) Hand off to torch as a zero-copy view of that GPU memory. Clone so
    #    the tensors outlive the engine without holding its pool open.
    tensors = persisted_to_torch(result, engine=engine)
    feat, label = tensors["feat"].clone(), tensors["label"].clone()
    del tensors

# 3) Train normally with DDP.
model = DDP(MyModel().cuda(), device_ids=[0])
train(model, feat, label)

dist.destroy_process_group()
```

What each step does:

1. `SPMDEngine.from_torch_distributed()` reads `(rank, world_size)` from
   the active torch process group, exchanges UCXX bootstrap addresses
   over `dist.broadcast_object_list`, and constructs an `SPMDEngine`
   bound to the resulting UCXX communicator. The engine is owned by
   the `with` block; closing the block releases the UCXX comm.
2. `engine.execute(lf)` returns a `PersistedQueryResult` whose partition
   stays GPU-resident on the rank that produced it.
   `persisted_to_torch(result, engine=engine)` views each column as a
   tensor sharing that GPU memory, so nothing round-trips through host
   memory. It consumes the partition, so the result cannot also be
   collected.
3. The training loop is a standard DDP loop. The ETL ranks and the DDP
   ranks are the same processes, so the per-rank tensors are exactly
   what DDP expects to see.

The handoff returns ordinary per-rank tensors rather than a `DTensor`.
See "Future work: DTensor" at the end for why, and for what it would
take to add one.

---

## Reading input

Nothing torch-specific applies here. File-based sources (`scan_parquet`,
`scan_csv`, ...) are partitioned automatically so that each rank reads a
different file or row-group range, and in-memory frames are already
rank-local. Pass every rank the *same* paths and let the engine split
them; do not hand each rank its own file list.

One caveat when testing: a dataset small enough to fit in a single
partition is read by a single rank, so a toy dataset will not exercise
multi-rank IO even though the results are correct.

See the [SPMD engine guide](../../../docs/cudf/source/cudf_polars/spmd_engine.md)
for the SPMD fundamentals this builds on: scan partitioning, the query
symmetry requirement, and collecting rank-local results.

---

## API reference

All entry points live under `cudf_polars.engine`.

### `SPMDEngine.from_torch_distributed`

```python
@classmethod
def from_torch_distributed(
    cls,
    *,
    group: "torch.distributed.ProcessGroup | None" = None,
    options: StreamingOptions | None = None,
    rapidsmpf_options: Options | None = None,
    executor_options: dict[str, Any] | None = None,
    engine_options: dict[str, Any] | None = None,
) -> SPMDEngine
```

Build an `SPMDEngine` that piggybacks on an already-initialized
`torch.distributed` process group.

**Preconditions:**
* `dist.is_initialized()` returns `True` on every rank.
* The caller has set the CUDA device for this rank
  (`torch.cuda.set_device(LOCAL_RANK)`).

**Behavior:**
* Rank 0 creates the UCXX root communicator and broadcasts its address
  through `dist.broadcast_object_list`.
* Non-root ranks construct their UCXX communicators from the
  broadcast address.
* All ranks barrier on the UCXX comm before the engine is returned.
* The engine owns the UCXX comm and closes it on `shutdown()` / `with`
  exit.

### `persisted_to_torch`

```python
def persisted_to_torch(
    result: PersistedQueryResult,
    *,
    engine: SPMDEngine,
    columns: list[str] | None = None,
) -> dict[str, torch.Tensor]
```

Convert this rank's GPU-resident result of `engine.execute(lf)` to a
dict of `torch.Tensor`, one per column. Each tensor is a zero-copy view
of the column's GPU memory, so the data never leaves the device, and the
tensors keep that memory alive.

Consumes the rank-local partition, so `result` cannot also be collected.
Columns must be fixed-width (numeric or boolean) and null-free; a string
column or a column containing nulls raises with a message saying what to
cast or fill.

The tensors hold the engine's GPU pool open for as long as they live.
`.clone()` them if they should outlive the engine.

### `polars_to_tensor`

```python
def polars_to_tensor(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]
```

Convert a Polars DataFrame to a dict of `torch.Tensor`, one per column.
Uses `pl.Series.to_torch()` under the hood. If `device` is provided,
each tensor is moved to that device. Per-column dtype overrides may be
supplied via `dtype`.

## Operational guidance

### Launching

Use `torchrun`, not `rrun`:

```
torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) script.py
```

`torchrun` sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and
`MASTER_PORT`. `SPMDEngine.from_torch_distributed()` reads rank and
world size from the torch process group; the rest of the env vars are
consumed by torch internally.

### Device placement

This is the one place the integration departs from normal
`torch.distributed` practice. The usual `torchrun` idiom leaves every
GPU visible and selects one by index, with
`torch.cuda.set_device(LOCAL_RANK)`. That does not work here, because
the engine runs on CUDA device ordinal 0, so its GPU has to come first
in `CUDA_VISIBLE_DEVICES`. `use_gpu` does that:

```python
from cudf_polars.engine.spmd import use_gpu

use_gpu(int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(0)
```

Call it *before* any CUDA call, including `dist.init_process_group(backend="nccl")`,
since initializing NCCL is itself one. After this the process sees exactly
one GPU, numbered 0, so use `device="cuda:0"` for tensor placement and
`device_ids=[0]` when wrapping a model in `DistributedDataParallel`.

Skipping this step is not a performance question. Constructing an engine
on a rank whose current device is not ordinal 0 raises, and without the
check it would have failed later with a CUDA error from deep inside
libcudf. See the [SPMD engine guide](../../../docs/cudf/source/cudf_polars/spmd_engine.md)
for why.

### Backend choice

Use `backend="nccl"` for the torch process group. NCCL is what DDP
expects for training collectives. The torch group is *only* used for
rendezvous (broadcasting the UCXX address) and for training; cudf-polars
collectives go through UCXX, which is built separately by the engine.

### Optional: share the RMM pool with torch

By default, cuDF/RMM and torch each manage their own GPU memory. For
workloads that stress the allocator (large joins next to large model
states), point torch at the same RMM pool:

```python
import rmm.allocators.torch
torch.cuda.memory.change_current_allocator(
    rmm.allocators.torch.rmm_torch_allocator
)
```

This is opinionated, touches torch global state, and is not enforced by
`from_torch_distributed`. Adopt it deliberately, not by default.

### Shutdown order

Tear down in the reverse order of setup:

```python
with SPMDEngine.from_torch_distributed() as engine:
    ...  # ETL + handoff

# ... training ...

dist.destroy_process_group()
```

Exiting the `with` block releases the UCXX comm, the RapidsMPF context,
and the thread pool. `dist.destroy_process_group()` then releases the
torch process group and NCCL communicators.

---

## Pitfalls

### Query symmetry

All ranks must issue the same sequence of Polars queries in the same
order. Collective operations are matched by a monotonically increasing
op id; a rank-conditional `collect`, early exit, or any branch that
causes different ranks to execute different query graphs will deadlock.
This is the same restriction that applies to plain `SPMDEngine` use.

### Do not mix launchers

Pick one of `torchrun` or `rrun`. The two launchers use different rank
discovery mechanisms, and `SPMDEngine.from_torch_distributed()` assumes
torch owns the rank assignment. If you launched with `rrun`, use plain
`SPMDEngine()` (which auto-bootstraps over UCXX) and skip the torch
integration.

### Tensors pin the engine's memory pool

`persisted_to_torch` returns zero-copy views into the engine's RMM pool,
so the pool cannot be released while they are alive. `.clone()` the
tensors you want to keep for training and drop the views.

### Rank numbering

`engine.rank` is the UCXX communicator's own numbering, assigned by the
order ranks connect, and it need not equal `dist.get_rank()`. This is
not specific to torch: under `rrun` it need not equal `RRUN_RANK`
either, and a four rank job was observed with `RRUN_RANK=1` holding
UCXX rank 2.

Nothing breaks, because each rank stores and reads its own data under
its own engine rank, and gradient all-reduce does not care which rank
holds which rows. What does break is code that treats the two
numberings as one, such as choosing input files by `dist.get_rank()`
and then assuming the engine labelled that rank's data the same way, or
naming output files by one and reading them back by the other. Pick one
numbering per purpose and stay with it.

---

## Future work: DTensor

The handoff returns ordinary per-rank tensors. Wrapping them as a
`DTensor`, so a query result looks like one logical tensor with
collectives inserted automatically, would be the natural next step. It
is left out because the layouts this engine produces are not ones
`DTensor` can describe.

`Shard(0)` does not mean "the rows are split across ranks somehow". The
[PyTorch documentation](https://docs.pytorch.org/docs/2.9/distributed.tensor.html)
states that it "follows the `torch.chunk(dim)` semantic", so for a given
global row count and rank count there is exactly one legal split: 100
rows over 8 ranks must be `[13, 13, 13, 13, 13, 13, 13, 9]`. This
engine splits by where the data falls. A hash-partitioned group-by over
4 ranks was measured at `[90602, 90749, 90236, 90794]`, where the only
legal split is `[90596, 90596, 90596, 90593]`.

The `shape` and `stride` arguments of `DTensor.from_local` do not bridge
that. They set the reported global shape, but torch re-derives each
rank's size from it with the same chunk rule, so the declared layout and
the real one still disagree. Supplying a shape for a `[3, 5]` split over
two ranks gave a correct `sum` but a wrong `mean`, 1.5 against 1.625,
and `full_tensor()` raised. Both are known PyTorch issues,
[#110762](https://github.com/pytorch/pytorch/issues/110762) and
[#144109](https://github.com/pytorch/pytorch/issues/144109), and the
documentation is explicit that a local tensor which is not a valid shard
leaves the resulting `DTensor` undefined.

A replicated result is representable, but a `DTensor` earns little
there: when every rank already holds a full copy, a global mean is a
local mean and there is no collective to insert.

Three ways this could land later:

1. **Rebalance before wrapping.** Redistribute rows onto the chunk
   split, after which `Shard(0)` is true. This is a collective that
   moves data and changes which rows a rank trains on.
2. **Adopt a ragged placement.**
   [RaggedShard](https://github.com/pytorch/pytorch/issues/169320)
   proposes per-rank allocations that `torch.chunk` cannot express,
   which is exactly what is missing. It is an open proposal, so this is
   a matter of waiting rather than building.
3. **Emit a replicated `DTensor` only**, for callers composing with
   `DeviceMesh` code. Cheap, and the one case that works today.

The legacy `ShardedTensor` with `EnumerableShardingSpec` can enumerate
arbitrary shard offsets and sizes, so it is worth naming and dismissing:
it is deprecated in favour of `DTensor`, and by its own docstring it
"doesn't provide any Tensor like operations", so it offers nothing over
the plain tensors returned here.

---

## Sketch: applying the same pattern to TensorFlow

The integration shape (SPMD launcher + UCXX bootstrap + per-rank tensor
handoff + framework-native training loop) is not PyTorch-specific. The
same recipe can target `tf.distribute.MultiWorkerMirroredStrategy`
(MWMS). It is not implemented today; this section is a sketch of what
the wiring would look like.

What stays the same:

- The SPMDEngine itself, plus its UCXX shuffle / group_by / join nodes.
- The mental model: every rank runs the same script, owns rank-local
  data, and uses a single process group for both ETL coordination and
  training collectives.
- The `CUDA_VISIBLE_DEVICES` pinning trick so the rapidsmpf hardware
  binding lines up with the framework's device choice.

What changes:

- **Launcher.** TF has no `torchrun` equivalent. MWMS workers find
  each other via the `TF_CONFIG` env var (a JSON cluster spec) that
  the caller populates per process via Kubeflow, the TF operator, an
  MPI launcher, or a hand-rolled shell script. A
  `SPMDEngine.from_tf_cluster()` helper would read `task.index` and
  the worker list from `TF_CONFIG` for `rank` / `world_size`.
- **Address exchange.** TF's `tf.distribute` API does not expose a
  `broadcast_object_list` equivalent on the strategy, so the UCXX
  root address has to be exchanged through a side channel: a shared
  filesystem write, a small gRPC service, or `MPI_Bcast` if the
  launcher provides an MPI communicator.
- **Data handoff.** Instead of `persisted_to_torch`, the natural helper
  is `polars_to_tf(df, ...) -> dict[str, tf.Tensor]` (using
  `tf.experimental.dlpack.from_dlpack` for zero-copy), and/or
  `polars_to_tf_dataset(df, batch_size=...) -> tf.data.Dataset`,
  which is what MWMS prefers to consume.
- **Replica vs worker.** PyTorch DDP scripts typically map one
  process to one GPU. MWMS supports multiple replicas per worker
  process via MirroredStrategy underneath; to stay in the
  one-process-one-GPU shape, set
  `tf.config.set_visible_devices(...)` per process, analogous to the
  `CUDA_VISIBLE_DEVICES` trick used here.

Sketch of the target usage:

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with SPMDEngine.from_tf_cluster() as engine:        # not implemented
    df = lf.collect(engine=engine)
    ds = polars_to_tf_dataset(df, batch_size=...)   # not implemented
    ds = strategy.experimental_distribute_dataset(ds)

with strategy.scope():
    model = build_model()
    model.compile(...)
model.fit(ds, ...)
```

There is no plan to ship this today; the cuDF/RAPIDS GPU-DL ecosystem
is overwhelmingly PyTorch, and TF GPU workloads tend to use
single-host `MirroredStrategy` rather than MWMS. The section exists so
that anyone evaluating cudf-polars for a TF workload can see what the
work would look like without re-deriving it from the PyTorch
integration.
