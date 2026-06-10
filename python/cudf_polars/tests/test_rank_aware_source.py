# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import cloudpickle
import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal

from cudf_polars.streaming.rank_aware_source import RankAwareSource, _PartitionSource
from cudf_polars.testing.asserts import assert_ir_translation_raises

# The scan-source classes below are defined in this test module. When shipped to
# Ray/Dask workers they would otherwise be pickled by reference, and the worker
# cannot import this test module. Register the module for pickle-by-value so the
# class definitions travel with the instances. Ray vendors its own cloudpickle,
# so register on that too when Ray is installed.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
try:
    from ray import cloudpickle as ray_cloudpickle
except ImportError:
    pass
else:
    ray_cloudpickle.register_pickle_by_value(sys.modules[__name__])


class _Source(RankAwareSource):
    """
    Rank-aware scan source that emits ``df`` on rank 0 only.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(
        self, with_columns, predicate, n_rows, batch_size, rank=None, nranks=None
    ):
        # Emit on the lone worker (rank is None) or on rank 0; every other rank
        # emits an empty same-schema frame so the union reconstructs ``df``.
        out = self.df.clear() if rank else self.df
        if with_columns is not None:
            out = out.select(with_columns)
        if predicate is not None:
            out = out.filter(predicate)
        if n_rows is not None:
            out = out.head(n_rows)
        yield out


def test_rank_aware_source_basic(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": pl.Series([1, 2, 3], dtype=pl.Int8())})
    q = register_io_source(_Source(df), schema={"a": pl.Int8})
    assert_frame_equal(q.collect(engine=engine), df)


def test_rank_aware_source_projection_pushdown(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    q = register_io_source(_Source(df), schema={"a": pl.Int64, "b": pl.Int64}).select(
        "a"
    )
    assert_frame_equal(q.collect(engine=engine), df.select("a"))


def test_rank_aware_source_predicate_pushdown(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    q = register_io_source(_Source(df), schema={"a": pl.Int64}).filter(pl.col("a") > 2)
    assert_frame_equal(q.collect(engine=engine), df.filter(pl.col("a") > 2))


class _MultiBatchSource(RankAwareSource):
    """Rank-aware scan source that emits two batches on rank 0 only."""

    def __init__(self, batches: list[pl.DataFrame]) -> None:
        self.batches = batches

    def __call__(
        self, with_columns, predicate, n_rows, batch_size, rank=None, nranks=None
    ):
        # Emit on the lone worker (rank is None) or on rank 0 only.
        if rank:
            yield from (b.clear() for b in self.batches)
        else:
            yield from self.batches


def test_rank_aware_source_multi_batch(engine: pl.GPUEngine):
    batches = [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]})]
    q = register_io_source(_MultiBatchSource(batches), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), pl.DataFrame({"a": [1, 2, 3, 4]}))


class _GpuSource(RankAwareSource):
    """Rank-aware source that yields GPU-resident cudf-polars DataFrames."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(
        self, with_columns, predicate, n_rows, batch_size, rank=None, nranks=None
    ):
        from cudf_polars.containers import DataFrame
        from cudf_polars.utils.cuda_stream import get_cuda_stream

        df = self.df if not rank else self.df.clear()
        yield DataFrame.from_polars(df, stream=get_cuda_stream())


def test_rank_aware_source_gpu_batches(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3]})
    q = register_io_source(_GpuSource(df), schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine), df)


class _PartitioningSource(RankAwareSource):
    """Self-partitions a shared frame across workers using ``rank``/``nranks``."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    def __call__(
        self, with_columns, predicate, n_rows, batch_size, rank=None, nranks=None
    ):
        if rank is None or nranks is None:
            out = self.df
        else:
            out = self.df.gather_every(nranks, offset=rank)
        if with_columns is not None:
            out = out.select(with_columns)
        yield out


def test_rank_aware_source_self_partition(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": list(range(10))})
    q = register_io_source(_PartitioningSource(df), schema={"a": pl.Int64})
    # Order across ranks is not guaranteed, so compare after sorting.
    assert_frame_equal(q.collect(engine=engine).sort("a"), df)


def test_rank_aware_source_from_loaders(engine: pl.GPUEngine):
    # Loaders are zero-argument callables, resolved lazily per worker.
    loaders = {
        0: lambda: pl.DataFrame({"a": [1, 2]}),
        1: lambda: pl.DataFrame({"a": [3, 4]}),
    }
    q = RankAwareSource.from_loaders(loaders, schema={"a": pl.Int64})
    expected = pl.DataFrame({"a": [1, 2, 3, 4]})
    assert_frame_equal(q.collect(engine=engine).sort("a"), expected)


def test_rank_aware_source_from_loaders_limit(engine: pl.GPUEngine):
    """A pushed-down ``.limit(n)`` on a PythonScan is rejected during translation."""
    loaders = {
        0: lambda: pl.DataFrame({"a": [1, 2]}),
        1: lambda: pl.DataFrame({"a": [3, 4]}),
    }
    q = RankAwareSource.from_loaders(loaders, schema={"a": pl.Int64}).limit(1)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_rank_aware_source_from_loaders_limit_default_engine():
    """The default Polars engine pushes ``.limit()`` into the source.

    ``from_loaders`` cannot honor a row limit across loaders, so it raises
    rather than over-returning rows.
    """
    loaders = {
        0: lambda: pl.DataFrame({"a": [1, 2]}),
        1: lambda: pl.DataFrame({"a": [3, 4]}),
    }
    q = RankAwareSource.from_loaders(loaders, schema={"a": pl.Int64}).limit(1)
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect()


def test_rank_aware_source_from_loaders_unreachable_rank():
    # The loader for rank 2 cannot be emitted by any worker when nranks == 2.
    source = _PartitionSource(
        {0: lambda: pl.DataFrame({"a": [1]}), 2: lambda: pl.DataFrame({"a": [2]})},
        {"a": pl.Int64},
    )
    with pytest.raises(ValueError, match="no worker can emit"):
        list(source(None, None, None, None, rank=0, nranks=2))


def test_rank_aware_source_from_loaders_gpu(engine: pl.GPUEngine):
    # Loaders returning GPU-resident cudf-polars DataFrames (zero-copy).
    def make_loader(values: list[int]):
        def load():
            from cudf_polars.containers import DataFrame
            from cudf_polars.utils.cuda_stream import get_cuda_stream

            return DataFrame.from_polars(
                pl.DataFrame({"a": values}), stream=get_cuda_stream()
            )

        return load

    loaders = {0: make_loader([1, 2]), 1: make_loader([3, 4])}
    q = RankAwareSource.from_loaders(loaders, schema={"a": pl.Int64})
    expected = pl.DataFrame({"a": [1, 2, 3, 4]})
    assert_frame_equal(q.collect(engine=engine).sort("a"), expected)


def _plain_source(with_columns, predicate, n_rows, batch_size):
    yield pl.DataFrame({"a": [1]})


def _scan_fn(source) -> object:
    """The register_io_source wrapper stored in the PythonScan options."""
    lf = register_io_source(source, schema={"a": pl.Int64})
    return lf._ldf.visit().view_current_node().options[0]


class _LowerState:
    """Minimal stand-in for the lowering transformer (only ``state`` is read)."""

    def __init__(self, nranks: int) -> None:
        self.state = {"nranks": nranks}


def _lower_python_scan(source, nranks: int) -> int:
    import cudf_polars.streaming.io  # noqa: F401  (registers the lower handler)
    from cudf_polars.containers import DataType
    from cudf_polars.dsl import ir
    from cudf_polars.streaming.dispatch import lower_ir_node

    schema = {"a": DataType(pl.Int64())}
    # options = (scan_fn, with_columns, source_type)
    node = ir.PythonScan(schema, (_scan_fn(source), None, "io_plugin"), None)
    # _LowerState is a minimal stub, not a full GenericTransformer.
    _, partition_info = lower_ir_node(node, _LowerState(nranks=nranks))  # type: ignore[arg-type]
    return partition_info[node].count


def test_python_scan_partition_count():
    # On a multi-rank run a PythonScan is a partition per rank, so downstream
    # global operators insert the required reduction; single-rank is one partition.
    assert _lower_python_scan(_plain_source, nranks=3) == 3
    assert _lower_python_scan(_plain_source, nranks=1) == 1
