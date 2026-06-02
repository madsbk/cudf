# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

from rmm.pylibrmm.stream import Stream

import pylibcudf as plc


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
    ],
)
def test_table_shape(arrow_tbl):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)

    assert plc_tbl.shape() == arrow_tbl.shape


@pytest.mark.parametrize("stream", [None, Stream()])
def test_table_to_arrow(table_data, stream):
    plc_tbl, _ = table_data
    expect = plc_tbl.tbl
    got = expect.to_arrow(stream=stream)
    # The order of `got` and `expect` is reversed here
    # because in almost all pylibcudf tests the `expect`
    # is a pyarrow object while `got` is a pylibcudf object,
    # whereas in this case those types are reversed.
    assert_table_eq(got, expect)


def test_table_copy(table_data):
    plc_tbl, _ = table_data
    original = plc_tbl.tbl

    copied = original.copy()

    assert copied.shape() == original.shape()
    assert copied.num_columns() == original.num_columns()
    assert copied.num_rows() == original.num_rows()

    for orig_col, copy_col in zip(
        original.columns(), copied.columns(), strict=True
    ):
        assert orig_col is not copy_col

    assert_table_eq(original.to_arrow(), copied)


def test_zero_column_table_num_rows_override():
    # A table with zero columns can carry an explicit row count.
    # See https://github.com/rapidsai/cudf/issues/21428
    tbl = plc.Table([], num_rows=5)
    assert tbl.num_columns() == 0
    assert tbl.num_rows() == 5
    assert tbl.shape() == (5, 0)
    # The override is preserved by copy().
    assert tbl.copy().num_rows() == 5


def test_zero_column_table_default_num_rows():
    # Without an override a zero-column table still reports zero rows.
    assert plc.Table([]).num_rows() == 0


def test_zero_column_table_num_rows_mismatch_raises():
    col = plc.Column.from_arrow(pa.array([1, 2, 3]))
    with pytest.raises(ValueError):
        plc.Table([col], num_rows=4)


def test_zero_column_table_concatenate_sums_rows():
    # Concatenating zero-column tables sums their row counts end-to-end.
    # See https://github.com/rapidsai/cudf/issues/21428
    result = plc.concatenate.concatenate(
        [plc.Table([], num_rows=7), plc.Table([], num_rows=5)]
    )
    assert result.num_columns() == 0
    assert result.num_rows() == 12


def test_from_arrow_zero_column_preserves_num_rows():
    # The reported reproduction for https://github.com/rapidsai/cudf/issues/21428:
    # a zero-width frame (rows but no columns) must round-trip its row count
    # through from_arrow. polars is used because pyarrow cannot construct a
    # zero-column / N-row batch.
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(height=5)
    assert df.shape == (5, 0)

    tbl = plc.Table.from_arrow(df)
    assert tbl.num_columns() == 0
    assert tbl.num_rows() == 5
