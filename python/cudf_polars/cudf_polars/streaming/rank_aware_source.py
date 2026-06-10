# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Rank-aware Python scan source for cudf-polars."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import polars as pl
from polars.io.plugins import register_io_source

from cudf_polars.containers import DataFrame

Loader: TypeAlias = Callable[[], pl.DataFrame | DataFrame]

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from polars._typing import SchemaDict


class RankAwareSource(abc.ABC):
    """
    Base class for Polars IO-plugin sources that can partition work by rank.

    Subclasses implement :meth:`__call__` and pass an instance to
    :func:`polars.io.plugins.register_io_source`. The call signature is the
    standard Polars ``io_source`` contract plus ``rank`` and ``nranks`` keyword
    arguments supplied by cudf-polars streaming engines.

    During multi-rank streaming execution every rank calls the source. The
    implementation must use ``rank`` and ``nranks`` to emit only rank-local rows
    so that the union across ranks reconstructs the dataset exactly once.

    During in-memory cudf-polars execution and default Polars execution, there
    is no streaming rank context and both values are ``None``.

    Examples
    --------
    Register a source that stripes a shared frame across streaming ranks.

    >>> import polars as pl
    >>> from polars.io.plugins import register_io_source
    >>> from cudf_polars.streaming.rank_aware_source import RankAwareSource
    >>>
    >>> class StripedSource(RankAwareSource):
    ...     def __init__(self, df):
    ...         self.df = df
    ...
    ...     def __call__(
    ...         self,
    ...         with_columns,
    ...         predicate,
    ...         n_rows,
    ...         batch_size,
    ...         rank=None,
    ...         nranks=None,
    ...     ):
    ...         if rank is None or nranks is None:
    ...             out = self.df
    ...         else:
    ...             out = self.df.gather_every(nranks, offset=rank)
    ...         if with_columns is not None:
    ...             out = out.select(with_columns)
    ...         if predicate is not None:
    ...             out = out.filter(predicate)
    ...         yield out
    >>>
    >>> lf = register_io_source(
    ...     StripedSource(pl.DataFrame({"a": [1, 2, 3]})),
    ...     schema={"a": pl.Int64},
    ... )

    See Also
    --------
    from_loaders : Build an IO source from a ``{rank: loader}`` mapping
        without writing rank-routing or empty-frame boilerplate by hand.
    """

    @abc.abstractmethod
    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int | None = None,
        nranks: int | None = None,
    ) -> Iterator[pl.DataFrame | DataFrame]:
        """
        Produce the rank-local batches of this source.

        Parameters
        ----------
        with_columns
            Projected column names. If not ``None``, the source should return
            only these columns.
        predicate
            Polars expression. The reader must filter their rows
            accordingly.
        n_rows
            Maximum number of rows requested from the scan. The cudf-polars
            engine does not support this and raises if not None.
        batch_size
            Optional hint for the number of rows to yield per chunk.
        rank
            Rank running this scan function, bound by the streaming engine.
            ``None`` for single-rank / in-memory / default Polars-engine
            execution.
        nranks
            Total number of ranks (the world size), bound by the streaming
            engine. ``None`` for single-rank execution.

        Returns
        -------
        Chunks for this rank.

        Notes
        -----
        A GPU-resident `cudf_polars.containers.DataFrame` can only be
        consumed by a cudf-polars engine; the default (host) Polars engine
        cannot. The source is not told which engine is collecting it, so a
        source that yields GPU frames is implicitly restricted to cudf-polars
        engines. Yield :class:`polars.DataFrame` if the source must also work
        with the default Polars engine.

        The emitted columns (after applying ``with_columns``) must match the
        registered schema in name, order, and dtype. cudf-polars validates this
        and raises :class:`polars.exceptions.SchemaError` on a mismatch. Polars
        only performs this check when ``register_io_source`` is called with
        ``validate_schema=True``, but that flag is not exposed to the GPU plan,
        so cudf-polars always validates.
        """

    @classmethod
    def from_loaders(
        cls,
        loaders: Mapping[int, Loader],
        *,
        schema: SchemaDict,
    ) -> pl.LazyFrame:
        """
        Build a IO source from rank-indexed partition loaders.

        This covers the common "I have one loader per rank" case.

        Parameters
        ----------
        loaders
            Mapping from rank to a zero-argument loader producing that rank's
            partition. A loader is invoked lazily, only on the rank that owns it,
            so remote or expensive partitions are not materialized to the caller
            up front. The loader may return a host :class:`polars.DataFrame` or a
            GPU-resident `cudf_polars.containers.DataFrame` (the latter requires a
            cudf-polars engine; see :meth:`__call__`).
        schema
            Output schema, used both for registration and to build the empty
            (host) frame emitted by ranks with no partition.

        Returns
        -------
        IO source backed by the rank specific loaders.

        Notes
        -----
        A loader is invoked lazily, during collection, inside the cudf-polars
        streaming runtime's own executor thread on the consuming rank. It does
        not run on the main thread or in a framework-managed task context.
        Loader authors must account for this environment:

        - **It runs in a plain thread.** Engine task/rank thread-locals are not
          set there; for example, ``distributed.get_worker()`` raises
          ``ValueError`` even on a Dask rank because the streaming executor
          thread is not a Dask task thread. Anything the loader needs about its
          location must be captured in its closure or bound arguments.
        - **It must be self-contained and picklable.** The loader travels inside
          the IR to the rank, so it cannot close over unpicklable state. A
          :func:`functools.partial` over a module-level function works well; a
          lambda over a local may not survive every serializer.
        - **It may also run in the caller's process.** On the single-rank path
          (``rank is None`` / ``nranks == 1``, e.g. the default Polars engine)
          every loader is invoked in the collecting process, so a loader that
          fetches from a remote rank must handle "not local" as well as "local".

        A pushed-down row limit (``.head(n)`` / ``.limit(n)``) on a PythonScan is
        not supported by the cudf-polars engine and is rejected during translation.
        """
        return register_io_source(
            _PartitionSource(dict(loaders), schema),  # type: ignore[arg-type]
            schema=schema,
        )


class _PartitionSource(RankAwareSource):
    """
    Rank-aware source backed by one optional loader per rank.

    Parameters
    ----------
    loaders
        Mapping from rank to a zero-argument partition loader.
    schema
        Output schema used for registration and empty frames on ranks without a
        loader.
    """

    def __init__(self, loaders: Mapping[int, Loader], schema: SchemaDict) -> None:
        self._loaders = loaders
        self._schema = schema

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
        rank: int | None = None,
        nranks: int | None = None,
    ) -> Iterator[pl.DataFrame | DataFrame]:
        if n_rows is not None:
            # cudf-polars rejects a pushed-down row limit during translation, so
            # this only happens under the default Polars engine. Honoring it
            # would require a global row budget across loaders; raise rather than
            # silently over-returning rows.
            raise NotImplementedError(
                "from_loaders does not support a pushed-down row limit "
                "(head/limit/tail)."
            )
        if nranks is not None and nranks > 1:
            unreachable = sorted(r for r in self._loaders if not 0 <= r < nranks)
            if unreachable:
                raise ValueError(
                    f"from_loaders has partitions for ranks {unreachable} that no "
                    f"worker can emit at world size nranks={nranks}."
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
            yield empty.select(with_columns) if with_columns is not None else empty
            return
        for loader in loaders:
            frame = loader()
            if with_columns is not None:
                frame = frame.select(with_columns)
            if predicate is not None:
                # A predicate is only ever delivered with a host dataframe: on
                # the cudf-polars path it is applied downstream on the device (so
                # it is None for any GPU dataframe), and the default Polars engine
                # cannot consume a GPU dataframe at all. So the frame is always a
                # pl.DataFrame here and we can filter it directly.
                assert isinstance(frame, pl.DataFrame)
                frame = frame.filter(predicate)
            yield frame
