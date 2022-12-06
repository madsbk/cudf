# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import threading
import weakref
from contextlib import ContextDecorator
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.buffer.spillable_buffer import SpillableBuffer, SpillLock


def as_buffer(
    data: Union[int, Any],
    *,
    size: int = None,
    owner: object = None,
    exposed: bool = False,
) -> Buffer:
    """Factory function to wrap `data` in a Buffer object.

    If `data` isn't a buffer already, a new buffer that points to the memory of
    `data` is created. If `data` represents host memory, it is copied to a new
    `rmm.DeviceBuffer` device allocation. Otherwise, the memory of `data` is
    **not** copied, instead the new buffer keeps a reference to `data` in order
    to retain its lifetime.

    If `data` is an integer, it is assumed to point to device memory.

    Raises ValueError if data isn't C-contiguous.

    Parameters
    ----------
    data : int or buffer-like or array-like
        An integer representing a pointer to device memory or a buffer-like
        or array-like object. When not an integer, `size` and `owner` must
        be None.
    size : int, optional
        Size of device memory in bytes. Must be specified if `data` is an
        integer.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.
    exposed : bool, optional
        Mark the buffer as permanently exposed (unspillable). This is ignored
        unless spilling is enabled and the data represents device memory, see
        SpillableBuffer.

    Return
    ------
    Buffer
        A buffer instance that represents the device memory of `data`.
    """

    if isinstance(data, Buffer):
        return data

    # We handle the integer argument in the factory function by wrapping
    # the pointer in a `__cuda_array_interface__` exposing object so that
    # the Buffer (and its sub-classes) do not have to.
    if isinstance(data, int):
        if size is None:
            raise ValueError(
                "size must be specified when `data` is an integer"
            )
        data = cuda_array_interface_wrapper(ptr=data, size=size, owner=owner)
    elif size is not None or owner is not None:
        raise ValueError(
            "`size` and `owner` must be None when "
            "`data` is a buffer-like or array-like object"
        )

    if get_global_manager() is not None:
        if hasattr(data, "__cuda_array_interface__"):
            return SpillableBuffer._from_device_memory(data, exposed=exposed)
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return SpillableBuffer._from_host_memory(data)

    if hasattr(data, "__cuda_array_interface__"):
        return Buffer._from_device_memory(data)
    return Buffer._from_host_memory(data)


_thread_spill_locks: Dict[int, Tuple[Optional[SpillLock], int]] = {}


def _push_thread_spill_lock() -> None:
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks.get(_id, (None, 0))
    if spill_lock is None:
        spill_lock = SpillLock()
    _thread_spill_locks[_id] = (spill_lock, count + 1)


def _pop_thread_spill_lock() -> None:
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks[_id]
    if count == 1:
        spill_lock = None
    _thread_spill_locks[_id] = (spill_lock, count - 1)


class acquire_spill_lock(ContextDecorator):
    """Decorator and context to set spill lock automatically.

    All calls to `get_spill_lock()` within the decorated function or context
    will return a spill lock with a lifetime bound to the function or context.

    Developer Notes
    ---------------
    We use the global variable `_thread_spill_locks` to track the global spill
    lock state. To support concurrency, each thread tracks its own state by
    pushing and popping from `_thread_spill_locks` using its thread ID.
    """

    def __enter__(self) -> Optional[SpillLock]:
        _push_thread_spill_lock()
        return get_spill_lock()

    def __exit__(self, *exc):
        _pop_thread_spill_lock()


def get_spill_lock() -> Union[SpillLock, None]:
    """Return a spill lock within the context of `acquire_spill_lock` or None

    Returns None, if spilling is disabled.
    """

    if get_global_manager() is None:
        return None
    _id = threading.get_ident()
    spill_lock, _ = _thread_spill_locks.get(_id, (None, 0))
    return spill_lock


class cached_property_delete_column_when_spilled(cached_property):
    """A version of `cached_property` that delete instead of spill the cache

    This property expect a `Column` instance.

    When cudf spilling is enabled, this property register a spill handler for
    the cached column that deletes the column rather than spilling it.
    See `SpillManager.register_spill_handler`.
    """

    def __get__(self, instance, owner=None):
        ret = super().__get__(instance, owner)
        manager = get_global_manager()
        if manager is None:
            return ret
        nbytes = ret.memory_usage
        assert isinstance(ret.base_data, SpillableBuffer)

        # We register a callback function `f` that clears the cache and
        # to avoid keeping `instance` alive, `f` takes a weak reference
        # of `instance`.
        def f(idx_ref: weakref.ReferenceType) -> int:
            idx = idx_ref()
            if idx is None:
                return 0
            idx.__dict__.pop(self.attrname, None)
            return nbytes

        manager.register_spill_handler(ret.base_data, f, weakref.ref(instance))
        return ret


if TYPE_CHECKING:
    from cudf._lib.column import Column


def get_columns(obj: Any) -> Set[Column]:
    """Return all columns in `obj` (no duplicates)"""

    from cudf._lib.column import Column
    from cudf.core.column_accessor import ColumnAccessor
    from cudf.core.frame import Frame
    from cudf.core.index import BaseIndex, RangeIndex
    from cudf.core.indexed_frame import IndexedFrame

    def _get_columns(obj: object, found: Set[Column]) -> None:
        if isinstance(obj, RangeIndex):
            return
        elif isinstance(obj, Column):
            found.add(obj)
        elif isinstance(obj, IndexedFrame):
            _get_columns(obj._data, found)
            _get_columns(obj._index, found)
        elif isinstance(obj, (Frame, BaseIndex)):
            _get_columns(obj._data, found)
        elif isinstance(obj, ColumnAccessor):
            for o in obj.columns:
                _get_columns(o, found)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                _get_columns(o, found)
        elif isinstance(obj, Mapping):
            for o in obj.values():
                _get_columns(o, found)

    ret: Set[Column] = set()
    _get_columns(obj, found=ret)
    return ret


def zeroing_column_offset_inplace(col: Column) -> None:
    mask = None if col.base_mask is None else col.mask
    children = None if col.base_children is None else col.children
    if col.data is not None:
        col.set_base_data(col.data)
    col._offset = 0
    if col.base_mask is not None:
        col.set_base_mask(mask)
    if col.base_children is not None:
        col.set_base_children(tuple(children))
