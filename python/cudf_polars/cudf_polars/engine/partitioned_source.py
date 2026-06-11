# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A rank-aware scan source over per-rank query-result partitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.io.plugins import register_io_source

from cudf_polars.containers import DataFrame
from cudf_polars.streaming.rank_aware_source import RankAwareSource, SizedChunks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from polars._typing import SchemaDict

    # A zero-argument loader returning this rank's partition (host or GPU frame).
    Loader = Callable[[], "pl.DataFrame | DataFrame"]


def _raising(exc: Exception) -> Iterator[pl.DataFrame | DataFrame]:
    """
    Return an iterator that re-raises ``exc`` when first advanced.

    Deferring a validation error to iteration time lets the default Polars
    engine surface it as a ``ComputeError`` (its handling for a failing IO
    source) rather than as a raw exception at plan-build time.
    """

    def gen() -> Iterator[pl.DataFrame | DataFrame]:
        raise exc
        yield  # pragma: no cover - unreachable; makes this a generator

    return gen()


def _height(frame: pl.DataFrame | DataFrame) -> int:
    """Row count of a host or GPU-resident frame."""
    return frame.num_rows if isinstance(frame, DataFrame) else frame.height


def _row_slice(
    frame: pl.DataFrame | DataFrame, offset: int, length: int
) -> pl.DataFrame | DataFrame:
    """Slice ``length`` rows from ``offset`` of a host or GPU-resident frame."""
    if isinstance(frame, DataFrame):
        return frame.slice((offset, length))
    return frame.slice(offset, length)


def _spans(
    frame: pl.DataFrame | DataFrame, max_rows_per_chunk: int | None
) -> Iterator[tuple[int, int]]:
    """
    Yield ``(offset, length)`` row-spans covering ``frame``.

    A zero-row frame still yields one (empty) span so the announced chunk count
    matches the chunks emitted.
    """
    nrows = _height(frame)
    if nrows == 0:
        yield 0, 0
        return
    step = max_rows_per_chunk or nrows
    for offset in range(0, nrows, step):
        yield offset, min(step, nrows - offset)


def _project(
    frame: pl.DataFrame | DataFrame,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
) -> pl.DataFrame | DataFrame:
    """Apply projection and (host-only) predicate pushdown to a chunk."""
    if with_columns is not None:
        frame = frame.select(with_columns)
    if predicate is not None:
        # A predicate is only ever delivered with a host dataframe: on the
        # cudf-polars path it is applied downstream on the device (so it is None
        # for any GPU dataframe), and the default Polars engine cannot consume a
        # GPU dataframe at all. So the frame is always a pl.DataFrame here.
        assert isinstance(frame, pl.DataFrame)
        frame = frame.filter(predicate)
    return frame


class PartitionedSource(RankAwareSource):
    """
    Rank-aware source backed by one optional loader per rank.

    Parameters
    ----------
    loaders
        Mapping from rank to a zero-argument loader producing that rank's
        partition. A loader is invoked lazily, only on the rank that owns it, so
        a remote partition is not materialized to the caller up front. The loader
        may return a host :class:`polars.DataFrame` or a GPU-resident
        :class:`~cudf_polars.containers.DataFrame`.
    schema
        Output schema, used to build the empty frame emitted by ranks without a
        loader.
    max_rows_per_chunk
        If set, each partition is sliced into chunks of at most this many rows and
        emitted as a :class:`SizedChunks` so the streaming engine keeps only one
        device chunk resident at a time. ``None`` emits each partition whole.
    """

    def __init__(
        self,
        loaders: Mapping[int, Loader],
        schema: SchemaDict,
        max_rows_per_chunk: int | None = None,
    ) -> None:
        self._loaders = loaders
        self._schema = schema
        self._max_rows_per_chunk = max_rows_per_chunk

    @classmethod
    def register(
        cls,
        loaders: Mapping[int, Loader],
        schema: SchemaDict,
        max_rows_per_chunk: int | None = None,
    ) -> pl.LazyFrame:
        """
        Build a :class:`PartitionedSource` and register it as a ``LazyFrame``.

        Parameters
        ----------
        loaders
            Mapping from rank to a zero-argument partition loader.
        schema
            Output schema for registration and empty-frame construction.
        max_rows_per_chunk
            Row-chunk size; ``None`` emits each partition whole.

        Returns
        -------
        A :class:`~polars.LazyFrame` backed by the per-rank partitions.
        """
        # register_io_source types io_source as a 4-arg callable returning host
        # frames; a RankAwareSource intentionally has extra (engine-bound)
        # rank/nranks args and may yield GPU frames.
        return register_io_source(
            cls(loaders, schema, max_rows_per_chunk),  # type: ignore[arg-type]
            schema=schema,
        )

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int | None = None,
        nranks: int | None = None,
    ) -> SizedChunks:
        """Emit this rank's partition as :class:`SizedChunks` (see :class:`RankAwareSource`)."""
        # Validation errors are raised lazily (see _raising) so the default
        # Polars engine reports them as ComputeError, matching its handling of
        # any failing IO source.
        if n_rows is not None:
            # cudf-polars rejects a pushed-down row limit during translation, so
            # this only happens under the default Polars engine. Honoring it
            # would require a global row budget across ranks.
            return SizedChunks(
                1,
                _raising(
                    NotImplementedError(
                        "PartitionedSource does not support a pushed-down row "
                        "limit (head/limit/tail)."
                    )
                ),
            )
        if nranks is not None and nranks > 1:
            unreachable = sorted(r for r in self._loaders if not 0 <= r < nranks)
            if unreachable:
                return SizedChunks(
                    1,
                    _raising(
                        ValueError(
                            f"PartitionedSource has partitions for ranks "
                            f"{unreachable} that no worker can emit at world size "
                            f"nranks={nranks}."
                        )
                    ),
                )

        if rank is None or nranks is None or nranks == 1:
            # Single-rank run: emit every partition (concatenated downstream).
            loaders = list(self._loaders.values())
        elif rank in self._loaders:
            loaders = [self._loaders[rank]]
        else:
            loaders = []
        if not loaders:
            # Absent rank: emit an empty same-schema (host) frame.
            empty = pl.DataFrame(schema=self._schema)
            empty = empty.select(with_columns) if with_columns is not None else empty
            return SizedChunks(1, iter((empty,)))

        # Fetch each selected partition once (one fetch on the rank-local path),
        # then plan row-slices so the chunk count is known up front. Reporting
        # the count lets the streaming engine emit one device chunk at a time.
        frames = [loader() for loader in loaders]
        spans = [
            (frame, offset, length)
            for frame in frames
            for offset, length in _spans(frame, self._max_rows_per_chunk)
        ]

        def chunks() -> Iterator[pl.DataFrame | DataFrame]:
            for frame, offset, length in spans:
                chunk = (
                    frame
                    if length == _height(frame)
                    else _row_slice(frame, offset, length)
                )
                chunk = _project(chunk, with_columns, predicate)
                if isinstance(chunk, DataFrame):
                    # A loader's GPU frame is typically still held by a
                    # persistent owner (dataframe_store / SPMDQueryResult), and
                    # slicing/projection produce zero-copy views of it. The
                    # streaming runtime wraps GPU chunks as exclusive_view=True
                    # (it may spill and account the device memory as freed), so
                    # emit an independently-owned deep copy to sever that sharing.
                    chunk = chunk.copy_deep()
                yield chunk

        return SizedChunks(len(spans), chunks())
