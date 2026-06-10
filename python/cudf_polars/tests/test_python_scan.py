# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io

import pytest

import polars as pl
from polars.io.plugins import register_io_source
from polars.testing import assert_frame_equal, assert_series_equal

from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_python_scan_function(engine_raise_on_fail: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3]})

    def source(with_columns, predicate, n_rows, batch_size):
        yield df

    q = register_io_source(source, schema={"a": pl.Int64})
    assert_frame_equal(q.collect(engine=engine_raise_on_fail), df)


def test_python_scan_function_multi_batch(engine_raise_on_fail: pl.GPUEngine):
    batches = [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]})]

    def source(with_columns, predicate, n_rows, batch_size):
        yield from batches

    q = register_io_source(source, schema={"a": pl.Int64})
    assert_frame_equal(
        q.collect(engine=engine_raise_on_fail), pl.DataFrame({"a": [1, 2, 3, 4]})
    )


def test_python_scan_empty_keeps_schema(engine_raise_on_fail: pl.GPUEngine):
    def source(with_columns, predicate, n_rows, batch_size):
        return iter(())

    q = register_io_source(source, schema={"a": pl.Int64})
    result = q.collect(engine=engine_raise_on_fail)
    assert result.shape == (0, 1)
    assert result.schema == pl.Schema({"a": pl.Int64})


def test_python_scan_predicate_pushdown(engine_raise_on_fail: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    def source(with_columns, predicate, n_rows, batch_size):
        yield df

    q = register_io_source(source, schema={"a": pl.Int64}).filter(pl.col("a") > 2)
    assert_frame_equal(
        q.collect(engine=engine_raise_on_fail), pl.DataFrame({"a": [3, 4, 5]})
    )


def test_python_scan_limit_unsupported(engine: pl.GPUEngine):
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    def source(with_columns, predicate, n_rows, batch_size):
        yield df

    q = register_io_source(source, schema={"a": pl.Int64}).limit(2)
    assert_ir_translation_raises(q, engine, NotImplementedError)


def test_python_scan_scalar_aggregate(engine: pl.GPUEngine):
    """A scalar aggregate over a plain source returns a single row on every engine.

    Regression: on a multi-rank run the source runs on rank 0 only, so the
    data-less ranks must not contribute a spurious aggregate row (e.g. an extra
    ``len == 0``); the result must match the CPU engine.
    """

    def source(with_columns, predicate, n_rows, batch_size):
        yield pl.DataFrame({"a": [1, 2, 3]})

    q = register_io_source(source, schema={"a": pl.Int64}).select(pl.len())
    assert_frame_equal(q.collect(engine=engine), q.collect())


def test_python_scan_groupby_aggregate(engine: pl.GPUEngine):
    def source(with_columns, predicate, n_rows, batch_size):
        yield pl.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})

    q = (
        register_io_source(source, schema={"a": pl.Int64, "b": pl.Int64})
        .group_by("a")
        .agg(pl.col("b").sum())
    )
    assert_frame_equal(q.collect(engine=engine).sort("a"), q.collect().sort("a"))


def test_python_scan_schema_mismatch(engine_raise_on_fail: pl.GPUEngine):
    def source(with_columns, predicate, n_rows, batch_size):
        # Float64 data, but the schema declares Int64.
        yield pl.DataFrame({"a": [1.0, 2.0, 3.0]})

    q = register_io_source(source, schema={"a": pl.Int64})
    with pytest.raises(pl.exceptions.SchemaError):
        q.collect(engine=engine_raise_on_fail)


def test_python_scan_schema_mismatch_nested(engine_raise_on_fail: pl.GPUEngine):
    def source(with_columns, predicate, n_rows, batch_size):
        yield pl.DataFrame({"a": [[1, 2], [3]]}, schema={"a": pl.List(pl.Int32)})

    q = register_io_source(source, schema={"a": pl.List(pl.Int64)})
    with pytest.raises(pl.exceptions.SchemaError):
        q.collect(engine=engine_raise_on_fail)


def test_python_scan_default_engine_fallback():
    df = pl.DataFrame({"a": [10, 20, 30]})

    def source(with_columns, predicate, n_rows, batch_size):
        yield df

    q = register_io_source(source, schema={"a": pl.Int64})
    assert_frame_equal(q.collect(), df)


def test_python_scan_predicate_and_projection(engine: pl.GPUEngine):
    """Filter + projection combos over a multi-batch source.

    Adapted from Polars:
    https://github.com/pola-rs/polars/blob/py-1.41.2/py-polars/tests/unit/io/test_plugins.py#L40
    """

    def source(with_columns, predicate, n_rows, batch_size):
        # A multi-batch source applying the pushed projection/predicate itself.
        # (On the cudf-polars path predicate is None -- it is applied on the GPU
        # -- and with_columns is honored here.)
        for i in [1, 2, 3]:
            df = pl.DataFrame({"a": [i], "b": [i]})
            if predicate is not None:
                df = df.filter(predicate)
            if with_columns is not None:
                df = df.select(with_columns)
            yield df

    q = register_io_source(source, schema={"a": pl.Int64, "b": pl.Int64})
    assert_frame_equal(
        q.collect(engine=engine).sort("a"),
        pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}),
    )
    assert_frame_equal(
        q.filter(pl.col("b") > 1).collect(engine=engine).sort("a"),
        pl.DataFrame({"a": [2, 3], "b": [2, 3]}),
    )
    assert_frame_equal(
        q.filter(pl.col("b") > 1).select("a").collect(engine=engine).sort("a"),
        pl.DataFrame({"a": [2, 3]}),
    )
    assert_frame_equal(
        q.select("a").collect(engine=engine).sort("a"),
        pl.DataFrame({"a": [1, 2, 3]}),
    )


def test_python_scan_reordered_columns(engine: pl.GPUEngine):
    """Reordered / projected columns from a source.

    Adapted from Polars:
    https://github.com/pola-rs/polars/blob/py-1.41.2/py-polars/tests/unit/io/test_io_plugin.py#L186
    """

    def source(with_columns, predicate, n_rows, batch_size):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [42, 13, 37]})
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    schema = {"a": pl.Int64, "b": pl.Int64}
    expected = pl.DataFrame({"b": [42, 13, 37], "a": [1, 2, 3]})
    assert_frame_equal(
        register_io_source(source, schema=schema)
        .select("b", "a")
        .collect(engine=engine)
        .sort("a"),
        expected,
    )


def test_python_scan_multi_chunk(engine: pl.GPUEngine):
    """A multi-chunk frame round-trips through the scan.

    Adapted from Polars (Int64 here, since cudf-polars does not support
    Categorical):
    https://github.com/pola-rs/polars/blob/py-1.41.2/py-polars/tests/unit/io/test_io_plugin.py#L287
    """
    df = pl.concat(
        [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]})],
        rechunk=False,
    )
    assert df.n_chunks() == 2

    def source(with_columns, predicate, n_rows, batch_size):
        yield df

    q = register_io_source(source, schema=df.schema)
    assert_frame_equal(q.collect(engine=engine).sort("a"), df)


def test_python_scan_lines(engine: pl.GPUEngine):
    """A batching line reader.

    Adapted from Polars:
    https://github.com/pola-rs/polars/blob/py-1.41.2/py-polars/tests/unit/io/test_io_plugin.py#L83
    """

    def scan_lines(f: io.BytesIO) -> pl.LazyFrame:
        schema = pl.Schema({"line": pl.String()})

        def generator(with_columns, predicate, n_rows, batch_size):
            if batch_size is None:
                batch_size = 100_000
            batch_lines: list[str] = []
            while n_rows != 0:
                batch_lines.clear()
                remaining = batch_size
                if n_rows is not None:
                    remaining = min(remaining, n_rows)
                    n_rows -= remaining
                while remaining != 0 and (line := f.readline().rstrip()):
                    batch_lines.append(line.decode())
                    remaining -= 1
                yield pl.Series("line", batch_lines, pl.String()).to_frame()
                if remaining != 0:
                    break

        return register_io_source(generator, schema=schema)

    text = "Hello\nThis is some text\nspread over multiple lines"
    f = io.BytesIO(text.encode())
    assert_series_equal(
        scan_lines(f).collect(engine=engine).to_series(),
        pl.Series("line", text.splitlines(), pl.String()),
    )


def test_python_scan_defer(engine: pl.GPUEngine):
    """`pl.defer` builds a PythonScan; it should run on the GPU engine."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    q = pl.defer(lambda: df, schema=df.schema)
    assert_frame_equal(q.collect(engine=engine).sort("a"), df)
