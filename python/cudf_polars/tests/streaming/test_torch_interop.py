# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch.distributed interop module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
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


@pytest.fixture
def single_rank_process_group(tmp_path: Path) -> Iterator[None]:
    """A real one-rank gloo process group, torn down afterwards."""
    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a torch.distributed process group is already initialized")
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("single_rank_process_group")
def test_from_torch_distributed_builds_engine_from_options() -> None:
    """The engine is built on the torch group and takes its options like `from_options`."""
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.spmd import SPMDEngine

    options = StreamingOptions(max_rows_per_partition=7)
    with SPMDEngine.from_torch_distributed(options) as engine:
        assert engine.nranks == 1
        assert engine.rank == 0
        executor = engine.config["executor_options"]
        assert executor["max_rows_per_partition"] == 7
