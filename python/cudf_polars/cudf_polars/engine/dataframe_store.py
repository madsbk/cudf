# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Process-local store of GPU-resident query-result partitions.

``execute()`` on the streaming engines keeps each rank's output partition
GPU-resident **in the process that produced it** (a Ray actor or a Dask worker,
both long-lived processes that own GPU memory), rather than copying it to host.
The partition is addressed by ``(query_id, rank)`` and read back on that same
process when the result is re-collected, so no data crosses a process boundary.

Because a GPU-resident :class:`~cudf_polars.containers.DataFrame` cannot be
serialized across processes, such a result is re-collectable only with the
engine that produced it (the same constraint as the SPMD engine).

The engine-specific piece is a small :class:`RetainedBackend` hook - "run this on
every rank-process and collect the ranks" - implemented over Ray actors and the
Dask client respectively. Everything else (the store, the scan loader, and the
:class:`RetainedQueryResult` lifetime machinery) is shared.
"""

from __future__ import annotations

import abc
import dataclasses
import weakref
from typing import TYPE_CHECKING

from cudf_polars.engine.core import drop_if_replicated, evaluate_on_rank
from cudf_polars.engine.partitioned_source import PartitionedSource

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable
    from concurrent.futures import ThreadPoolExecutor

    import polars as pl

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.ir import IR
    from cudf_polars.streaming.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


# Per-process registry of GPU-retained partitions, keyed by (query_id, rank).
# Each Ray actor / Dask worker process has its own. query_id is a uuid4, so keys
# are globally unique across engine instances sharing a process.
_STORE: dict[tuple[uuid.UUID, int], DataFrame] = {}


def put(query_id: uuid.UUID, rank: int, df: DataFrame) -> None:
    """Store this rank's partition on the current process."""
    _STORE[query_id, rank] = df


def get(query_id: uuid.UUID, rank: int) -> DataFrame:
    """Return this rank's partition from the current process."""
    return _STORE[query_id, rank]


def drop(query_id: uuid.UUID) -> None:
    """Drop every rank's partition for ``query_id`` on the current process (idempotent)."""
    for key in [k for k in _STORE if k[0] == query_id]:
        del _STORE[key]


def drop_many(query_ids: Iterable[uuid.UUID]) -> None:
    """Drop partitions for several queries on the current process (idempotent)."""
    wanted = set(query_ids)
    for key in [k for k in _STORE if k[0] in wanted]:
        del _STORE[key]


def clear() -> None:
    """
    Drop every partition on the current process (all queries; idempotent).

    Used when a process's streaming context is torn down (e.g. engine reset), so
    partitions allocated on the outgoing memory resource are freed before it goes
    away rather than outliving it.
    """
    _STORE.clear()


def store_partition(
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    query_id: uuid.UUID,
) -> int:
    """
    Evaluate ``ir`` on this rank and keep the GPU-retained result in the store.

    Runs the collective evaluation using this process's rapidsmpf context, drops
    a replicated output on non-root ranks, and stores the partition under
    ``(query_id, comm.rank)``. Returns this rank's index (nothing crosses the
    process boundary).
    """
    gpu_df, metadata = evaluate_on_rank(
        ctx, comm, py_executor, ir, config_options, query_id=query_id
    )
    gpu_df = drop_if_replicated(gpu_df, comm.rank, metadata)
    put(query_id, comm.rank, gpu_df)
    return comm.rank


@dataclasses.dataclass(frozen=True)
class DataFrameHandle:
    """
    Locator for one rank's retained partition.

    Only ``(query_id, rank)`` is needed: the scan loader runs on the owning
    process (rank affinity), so there is no address or object reference to carry.
    """

    query_id: uuid.UUID
    rank: int


class _DataFrameLoader:
    """
    Zero-argument loader that reads one rank's partition from the process store.

    Holds ``owner`` (the :class:`RetainedQueryResult`) so a ``LazyFrame`` derived
    from :meth:`RetainedQueryResult.lazy` keeps the result - and therefore its
    retained partitions - alive until that frame is dropped. ``owner`` is dropped
    when the loader is pickled into the IR for a worker/actor: the owning process
    only needs the handle to read its local store.
    """

    def __init__(self, handle: DataFrameHandle, owner: object = None) -> None:
        self._handle = handle
        self._owner = owner

    def __reduce__(self) -> tuple:
        # Strip ``owner`` on serialization (see class docstring).
        return (_DataFrameLoader, (self._handle, None))

    def __call__(self) -> DataFrame:
        return get(self._handle.query_id, self._handle.rank)


class RetainedBackend(abc.ABC):
    """
    Engine hook for running the produce/drop steps on every rank-process.

    Ray implements it over its actors; Dask over its client. Everything else in
    this module is engine-agnostic.
    """

    @abc.abstractmethod
    def execute_retained(
        self,
        ir: IR,
        config_options: ConfigOptions[StreamingExecutor],
        query_id: uuid.UUID,
    ) -> list[int]:
        """Run :func:`store_partition` on every rank-process; return the ranks."""

    @abc.abstractmethod
    def drop_retained(self, query_id: uuid.UUID) -> None:
        """Drop ``query_id``'s partitions on every rank-process (best-effort)."""


class RetainedQueryResult:
    """
    Distributed result whose partitions stay GPU-resident in their producing process.

    Returned by ``engine.execute()``. Re-collectable **only with the producing
    engine**: the partitions never leave their process, so the default (host)
    Polars engine cannot consume them.

    Partitions are released when this result (and every ``LazyFrame`` derived from
    :meth:`lazy`, which pins it alive) is garbage-collected, or eagerly via
    :meth:`release` / the context-manager protocol.

    Parameters
    ----------
    backend
        Engine hook used to drop the retained partitions.
    query_id
        Identifier of the query that produced the partitions.
    ranks
        Ranks that produced a partition.
    schema
        Output schema as a ``{column_name: polars_dtype}`` mapping.
    """

    def __init__(
        self,
        backend: RetainedBackend,
        query_id: uuid.UUID,
        ranks: list[int],
        schema: dict[str, pl.DataType],
    ) -> None:
        self._query_id = query_id
        self._ranks = ranks
        self._schema = schema
        self._finalizer = weakref.finalize(self, backend.drop_retained, query_id)

    def release(self) -> None:
        """
        Release the retained partitions now (idempotent).

        Invalidates any :class:`~polars.LazyFrame` previously returned by
        :meth:`lazy`; do not collect those afterwards.
        """
        self._finalizer()

    def __enter__(self) -> RetainedQueryResult:
        """Return self; :meth:`release` runs on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the retained partitions on scope exit."""
        self.release()

    def lazy(self) -> pl.LazyFrame:
        """
        Return a :class:`~polars.LazyFrame` backed by the retained partitions.

        Re-collecting with the producing engine runs the scan on each owning
        process, which reads its partition from the local store - no host
        round-trip. Whole GPU partitions are emitted (no row-chunking), since
        they are already device-resident.

        Returns
        -------
        LazyFrame with one partition per original rank.
        """
        loaders = {
            rank: _DataFrameLoader(DataFrameHandle(self._query_id, rank), self)
            for rank in self._ranks
        }
        return PartitionedSource.register(loaders, self._schema)
