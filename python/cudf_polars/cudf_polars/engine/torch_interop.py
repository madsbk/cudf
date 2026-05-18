# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Helpers to interoperate between ``SPMDEngine`` and ``torch.distributed``.

This module provides:

* :func:`persisted_to_torch` for handing the GPU-resident result of
  :meth:`~cudf_polars.engine.spmd.SPMDEngine.execute` to torch without a host
  round-trip. This is the recommended entry point.
* :func:`polars_to_tensor` for converting a host-side per-rank DataFrame,
  as returned by :meth:`~polars.LazyFrame.collect`, into tensors.

These return ordinary per-rank :class:`torch.Tensor` objects rather than
``DTensor``. A query result is split across ranks in counts that follow the
data, which is not the split ``torch.chunk`` makes, and that split is the only
one ``DTensor``'s ``Shard(0)`` can describe. Rank-local tensors are also what a
data-parallel training loop consumes. To slice a replicated result into per-rank
shards, ``tensor.chunk(world_size)[rank]`` is the whole operation.

``torch`` is imported lazily, so importing this module does not require a
``torch`` install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.unstable import unstable

if TYPE_CHECKING:
    import torch

    import polars as pl

    from cudf_polars.containers import Column, DataFrame
    from cudf_polars.engine.persisted_result import PersistedQueryResult
    from cudf_polars.engine.spmd import SPMDEngine


def _torch_dtypes() -> dict[Any, tuple[str, torch.dtype]]:
    """Map libcudf fixed-width type ids to a ``(numpy typestr, torch dtype)`` pair."""
    import torch

    import pylibcudf as plc

    tid = plc.types.TypeId
    return {
        tid.INT8: ("|i1", torch.int8),
        tid.INT16: ("<i2", torch.int16),
        tid.INT32: ("<i4", torch.int32),
        tid.INT64: ("<i8", torch.int64),
        tid.UINT8: ("|u1", torch.uint8),
        tid.UINT16: ("<u2", torch.uint16),
        tid.UINT32: ("<u4", torch.uint32),
        tid.UINT64: ("<u8", torch.uint64),
        tid.FLOAT32: ("<f4", torch.float32),
        tid.FLOAT64: ("<f8", torch.float64),
        tid.BOOL8: ("|b1", torch.bool),
    }


class _DeviceArrayView:
    """
    A ``__cuda_array_interface__`` view of a libcudf column's data buffer.

    Holds ``owner`` so the GPU memory outlives every tensor built from it.
    """

    def __init__(self, ptr: int, size: int, typestr: str, owner: object) -> None:
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "strides": None,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }
        self._owner = owner


def _column_to_tensor(name: str, column: Column, owner: object) -> torch.Tensor:
    """Zero-copy view of a fixed-width, non-nullable GPU column as a tensor."""
    import cupy
    import torch

    obj = column.obj
    type_id = obj.type().id()
    mapping = _torch_dtypes()
    if type_id not in mapping:
        raise TypeError(
            f"column {name!r} has dtype {column.dtype.polars_type}, which has no "
            "zero-copy torch equivalent. Cast it to a fixed-width numeric or "
            "boolean type before the handoff."
        )
    if obj.null_count() > 0:
        raise ValueError(
            f"column {name!r} contains nulls, which torch tensors cannot "
            "represent. Fill or drop them before the handoff, for example "
            f"with `pl.col({name!r}).fill_null(...)`."
        )
    typestr, dtype = mapping[type_id]
    buffer = obj.data()
    if buffer is None:  # empty column carries no data buffer
        return torch.empty(0, dtype=dtype, device="cuda")
    itemsize = torch.empty(0, dtype=dtype).element_size()
    view = _DeviceArrayView(
        buffer.ptr + obj.offset() * itemsize,
        obj.size(),
        typestr,
        (buffer, owner),
    )
    tensor = torch.as_tensor(cupy.asarray(view))
    # Anchor the source explicitly rather than relying on the refcount that
    # cupy/torch happen to keep through the array-interface handoff.
    tensor._cudf_polars_owner = view
    return tensor


def _dataframe_to_torch(
    df: DataFrame, columns: list[str] | None
) -> dict[str, torch.Tensor]:
    """Convert selected columns of a GPU DataFrame to tensors, zero-copy."""
    names = columns if columns is not None else df.column_names
    missing = [name for name in names if name not in df.column_map]
    if missing:
        raise KeyError(
            f"column(s) {missing} not in result; available: {df.column_names}"
        )
    return {name: _column_to_tensor(name, df.column_map[name], df) for name in names}


@unstable()
def persisted_to_torch(
    result: PersistedQueryResult,
    *,
    engine: SPMDEngine,
    columns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert this rank's GPU-resident query result to ``torch.Tensor`` objects.

    Takes the result of :meth:`~cudf_polars.engine.spmd.SPMDEngine.execute` and
    views each column as a tensor that shares the column's GPU memory, so the
    data never leaves the device. Contrast with
    :meth:`~polars.LazyFrame.collect`, which copies the result to host memory
    first.

    This consumes the rank-local partition (see
    :meth:`~cudf_polars.engine.persisted_result.PersistedQueryResult.take_local`),
    so ``result`` cannot also be collected. The returned tensors keep the
    underlying GPU memory alive.

    Parameters
    ----------
    result
        Result of ``engine.execute(lf)``.
    engine
        The engine that produced ``result``; supplies this process's rank.
    columns
        Subset of columns to convert; defaults to all columns of the result.

    Returns
    -------
    A dict mapping column name to a GPU :class:`torch.Tensor`.

    Raises
    ------
    KeyError
        If ``columns`` references a name that is not in the result.
    TypeError
        If a column's dtype has no zero-copy torch equivalent.
    ValueError
        If a column contains nulls.

    Examples
    --------
    >>> with SPMDEngine.from_torch_distributed() as engine:  # doctest: +SKIP
    ...     result = engine.execute(lf)
    ...     tensors = persisted_to_torch(result, engine=engine)
    """
    return _dataframe_to_torch(result.take_local(engine.rank), columns)


@unstable()
def polars_to_tensor(
    df: pl.DataFrame,
    *,
    columns: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert a Polars DataFrame to a dict of ``torch.Tensor``.

    Uses :meth:`polars.Series.to_torch` per column. If ``device`` is provided,
    each tensor is moved to that device after conversion. Per-column dtype
    overrides may be supplied via ``dtype``.

    Parameters
    ----------
    df
        Source DataFrame. Each column becomes one tensor.
    columns
        Subset of columns to convert; defaults to all columns of ``df``.
    device
        Optional target device (e.g. ``"cuda:0"`` or a :class:`torch.device`).
    dtype
        Optional per-column :class:`torch.dtype` overrides.

    Returns
    -------
    A dict mapping column name to :class:`torch.Tensor`.

    Raises
    ------
    KeyError
        If ``columns`` references a name that is not in ``df``.
    """
    cols = columns if columns is not None else df.columns
    dtype = dtype or {}

    out: dict[str, torch.Tensor] = {}
    for name in cols:
        series = df[name]  # raises KeyError if missing
        tensor = series.to_torch()
        target_dtype = dtype.get(name)
        if target_dtype is not None:
            tensor = tensor.to(dtype=target_dtype)
        if device is not None:
            tensor = tensor.to(device)
        out[name] = tensor
    return out
