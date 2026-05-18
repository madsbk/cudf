# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""
End-to-end example: distributed GPU ETL with cudf-polars, then DDP training.

Launch under torchrun with one process per GPU. The ``OMP_NUM_THREADS=1``
prefix silences torchrun's default-thread-count warning.

Single-GPU::

    OMP_NUM_THREADS=1 torchrun --nproc-per-node=1 \
        python/cudf_polars/examples/torch_distributed_etl.py

Multi-GPU::

    OMP_NUM_THREADS=1 torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) \
        python/cudf_polars/examples/torch_distributed_etl.py

What this example does, per rank:

1. Initializes the torch.distributed NCCL process group.
2. Builds an :class:`SPMDEngine` that piggybacks on that process group
   via :meth:`SPMDEngine.from_torch_distributed`.
3. Generates a small synthetic rank-local DataFrame, then runs two
   queries: a ``group_by("user_id")`` aggregate, whose result at this
   size comes back as a complete copy on every rank, and a row-wise
   projection of the rank-local training rows, which stays rank-local.
4. Trains a tiny MLP under :class:`DistributedDataParallel` on the
   sharded rows; gradients are all-reduced across ranks via NCCL.

Both queries use :meth:`SPMDEngine.execute`, which keeps each rank's
result GPU-resident, and the handoff to torch is a zero-copy view of
that GPU memory. Nothing round-trips through host memory.

The script is self-contained (no parquet files required) so it can run
on any multi-GPU box for smoke testing the integration.
"""

from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import polars as pl

from cudf_polars.engine.spmd import SPMDEngine, use_gpu
from cudf_polars.engine.torch_interop import persisted_to_torch

N_USERS = 100


def synthetic_data(rank: int, n_rows: int = 5_000) -> pl.LazyFrame:
    """
    Synthetic rank-local data: y ~= 2 * x + 1 + noise.

    Every rank draws ``user_id`` from the same global pool ``[0, N_USERS)``
    so the same id appears on several ranks, which is what makes the
    ``group_by`` a genuinely cross-rank reduction.
    """
    rng = random.Random(rank)
    xs = [rng.uniform(-3.0, 3.0) for _ in range(n_rows)]
    noise = [rng.gauss(0.0, 0.5) for _ in range(n_rows)]
    ys = [2.0 * x + 1.0 + n for x, n in zip(xs, noise, strict=True)]
    user_ids = [rng.randint(0, N_USERS - 1) for _ in range(n_rows)]
    return pl.LazyFrame({"user_id": user_ids, "feat": xs, "label": ys})


class TinyModel(nn.Module):
    """A small MLP used to exercise the DDP step."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.net(x)


def main() -> None:
    """Run the ETL + DDP example end to end."""
    # Give this rank its GPU before any CUDA work happens. torchrun, unlike
    # rrun and the Dask and Ray workers, leaves every GPU visible, and the
    # engine runs on ordinal 0. Afterwards the only visible device is 0.
    use_gpu(int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()

        with SPMDEngine.from_torch_distributed() as engine:
            raw = synthetic_data(rank)

            # ----- Query 1: global per-user aggregate -----
            # Each rank holds rows for many of the same users, so this
            # aggregate combines across ranks over UCXX. At this size the
            # result comes back as a complete copy on every rank, so the
            # statistics below are already global and need no collective.
            stats = engine.execute(
                raw.group_by("user_id").agg(
                    pl.col("feat").mean().cast(pl.Float32),
                    pl.len().alias("n"),
                )
            )
            replicated = stats.local_is_duplicated(engine.rank)
            feat = persisted_to_torch(stats, engine=engine, columns=["feat"])["feat"]
            if rank == 0:
                print(f"[rank 0] group_by output: replicated={replicated}")
                print(
                    f"[rank 0] {feat.numel()} users, "
                    f"global per-user feat mean: {feat.mean().item():.4f}"
                )

            # ----- Query 2: rank-local training rows -----
            # A row-wise projection, so each rank keeps its own disjoint
            # rows: this is the genuinely sharded output that DDP wants.
            # Had a query returned a replicated result we wanted to train
            # on instead, `tensor.chunk(world)[rank]` would slice it into
            # per-rank shards without any collective.
            train = engine.execute(
                raw.select(
                    pl.col("feat").cast(pl.Float32),
                    pl.col("label").cast(pl.Float32),
                )
            )
            sharded = not train.local_is_duplicated(engine.rank)
            tensors = persisted_to_torch(train, engine=engine)
            # These tensors are zero-copy views of the engine's GPU
            # memory, so they keep that memory reserved for as long as
            # they live. Clone to hand ownership to torch and let the
            # engine's pool be released at the end of the `with` block.
            x = tensors["feat"].clone().unsqueeze(1)
            y = tensors["label"].clone().unsqueeze(1)
            del tensors
            if rank == 0:
                print(f"\n[rank 0] training rows: sharded={sharded}")

        # ----- DDP training on the rank-local rows -----
        # Each rank trains on its own shard; DDP all-reduces gradients.
        model = TinyModel(in_features=1).to(device)
        ddp = DDP(model, device_ids=[0])
        opt = torch.optim.Adam(ddp.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        if rank == 0:
            print(f"Training tiny MLP under DDP across {world} rank(s):")
        for step in range(100):
            pred = ddp(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if rank == 0 and (step % 20 == 0 or step == 99):
                print(f"  step {step:>3d}  loss={loss.item():.4f}")

        # Cheap per-rank eval: pred/label correlation.
        with torch.no_grad():
            pred = ddp(x).squeeze(1)
            target = y.squeeze(1)
            corr = torch.corrcoef(torch.stack([pred, target]))[0, 1]
        print(
            f"[rank {rank}] {x.numel()} local rows, "
            f"pred/label correlation: {corr.item():.4f}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
