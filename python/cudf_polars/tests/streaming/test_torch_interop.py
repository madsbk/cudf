# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch.distributed interop module."""

from __future__ import annotations

import pytest

import polars as pl

torch = pytest.importorskip("torch")

from cudf_polars.engine.torch_interop import (  # noqa: E402
    polars_to_tensor,
)

# ---------------------------------------------------------------------------
# polars_to_tensor
# ---------------------------------------------------------------------------


def test_polars_to_tensor_all_columns() -> None:
    """All columns are converted by default and preserve values."""
    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": [7, 8, 9]},
    )
    out = polars_to_tensor(df)
    assert set(out) == {"a", "b", "c"}
    assert out["a"].tolist() == [1, 2, 3]
    assert out["b"].tolist() == [4.0, 5.0, 6.0]


def test_polars_to_tensor_column_subset() -> None:
    """`columns` selects a subset and preserves order."""
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = polars_to_tensor(df, columns=["c", "a"])
    assert list(out) == ["c", "a"]
    assert out["c"].tolist() == [5, 6]
    assert out["a"].tolist() == [1, 2]


def test_polars_to_tensor_dtype_override() -> None:
    """Per-column dtype overrides are applied."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = polars_to_tensor(df, dtype={"a": torch.float64})
    assert out["a"].dtype == torch.float64


def test_polars_to_tensor_missing_column_raises() -> None:
    """Requesting a column that does not exist raises."""
    df = pl.DataFrame({"a": [1, 2]})
    with pytest.raises((KeyError, pl.exceptions.ColumnNotFoundError)):
        polars_to_tensor(df, columns=["missing"])


def test_from_torch_distributed_requires_initialized_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classmethod raises if torch.distributed has not been initialized."""
    import torch.distributed as dist

    from cudf_polars.engine.spmd import SPMDEngine

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match=r"torch\.distributed is not initialized"):
        SPMDEngine.from_torch_distributed()


# ---------------------------------------------------------------------------


def test_from_torch_distributed_rejects_options_and_raw_mix() -> None:
    """Passing both `options` and the raw `*_options` triple is a TypeError."""
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.spmd import SPMDEngine

    opts = StreamingOptions()
    with pytest.raises(TypeError, match="either `options` or"):
        SPMDEngine.from_torch_distributed(options=opts, executor_options={})
