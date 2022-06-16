# Copyright (c) 2020-2022, NVIDIA CORPORATION.
from __future__ import annotations

import functools
import operator
import pickle
import time
from weakref import WeakSet
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable

if TYPE_CHECKING:
    from cudf._lib.column import AccessCounter
    from cudf.core.spill_manager import SpillManager


def format_bytes(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def get_base_buffer(obj: Any) -> Optional[Buffer]:
    if isinstance(obj, Buffer):
        if isinstance(obj._owner, rmm.DeviceBuffer):
            return obj
        return get_base_buffer(obj._owner)
    if hasattr(obj, "base_data"):
        return get_base_buffer(obj.base_data)
    return None


class Buffer(Serializable):
    """
    A Buffer represents a device memory allocation.

    Parameters
    ----------
    data : Buffer, array_like, int
        An array-like object or integer representing a
        device or host pointer to pre-allocated memory.
    size : int, optional
        Size of memory allocation. Required if a pointer
        is passed for `data`.
    owner : object, optional
        Python object to which the lifetime of the memory
        allocation is tied. If provided, a reference to this
        object is kept in this Buffer.
    sole_owner : bool, optional
        Whether or not this buffer is the sole owner of the
        underlying memory.
    ptr_exposed : bool, optional
        Whether or not a raw pointer (integer or C pointer) has
        been exposed to the outside world . If this is the case,
        the buffer cannot be spilled.
    """

    _ptr: Optional[int]
    _ptr_desc: dict
    _size: int
    _owner: object
    _access_counter: AccessCounter
    _ptr_exposed: bool
    _last_accessed: float
    _spill_manager: Optional[SpillManager]

    def __init__(
        self,
        data: Any = None,
        size: int = None,
        owner: object = None,
        ptr_exposed: bool = False,
    ):
        from cudf._lib.column import AccessCounter
        from cudf.core.spill_manager import global_manager

        self._lock = RLock()
        self._access_counter = AccessCounter()
        self._ptr_exposed = ptr_exposed
        self._ptr_desc = {"type": "gpu"}
        self._last_accessed = time.monotonic()
        self._viewers = WeakSet()

        if isinstance(data, Buffer):
            raise TypeError("Cannot do this")
        elif isinstance(data, rmm.DeviceBuffer):
            self._ptr = data.ptr
            self._size = data.size
            self._owner = data
        elif hasattr(data, "__array_interface__") or hasattr(
            data, "__cuda_array_interface__"
        ):
            self._init_from_array_like(data, owner)
        elif isinstance(data, memoryview):
            self._init_from_array_like(np.asarray(data), owner)
        elif isinstance(data, int):
            if not isinstance(size, int):
                raise TypeError("size must be integer")
            self._ptr = data
            self._size = size
            self._owner = owner
        elif data is None:
            self._ptr = 0
            self._size = 0
            self._owner = None
        else:
            try:
                data = memoryview(data)
            except TypeError:
                raise TypeError("data must be Buffer, array-like or integer")
            self._init_from_array_like(np.asarray(data), owner)

        if self._owner is not None:
            base_buffer = get_base_buffer(self._owner)
            if base_buffer is not None:
                with base_buffer._lock:
                    assert not base_buffer.is_spilled
            elif not isinstance(self._owner, rmm.DeviceBuffer):
                # the Buffer is constructed from externally owned memory,
                # so it can never be spilled:
                self._ptr_exposed = True

        self._spill_manager = None
        if global_manager.enabled:
            self._spill_manager = global_manager.get()
            if self._ptr and self._size and not self._ptr_exposed:
                base = self._spill_manager.lookup_address_range(
                    self._ptr, self._size
                )
                if base:
                    with base._lock:
                        if self is not base:
                            base._viewers.add(self)
                        return
                else:
                    if isinstance(self._owner, rmm.DeviceBuffer):
                        # TODO: find out why we even need this check
                        self._spill_manager.add(self)

    @classmethod
    def from_buffer(cls, buffer: Buffer, size: int = None, offset: int = 0):
        """
        Create a buffer from another buffer

        Parameters
        ----------
        buffer : Buffer
            The base buffer, which will also be set as the owner of
            the memory allocation.
        size : int, optional
            Size of the memory allocation (default: `buffer.size`).
        offset : int, optional
            Start offset relative to `buffer.ptr`.
        """
        ret = cls()
        with buffer._lock:
            buffer.move_inplace(target="gpu")
            ret._ptr = buffer._ptr + offset
            ret._size = buffer.size if size is None else size
            base = get_base_buffer(buffer)
            if base:
                with base._lock:
                    base._viewers.add(ret)
            ret._owner = base if base is not None else buffer
        return ret

    def __len__(self) -> int:
        return self._size

    @property
    def is_spilled(self) -> bool:
        return self._ptr_desc["type"] != "gpu"

    def move_inplace(self, target: str = "cpu") -> None:
        with self._lock:
            ptr_type = self._ptr_desc["type"]
            if ptr_type == target:
                return
            if (ptr_type, target) == ("gpu", "cpu"):
                if not self.spillable:
                    raise ValueError(
                        f"Cannot in-place move an unspillable buffer: {self}"
                    )
                base_buf = get_base_buffer(self)
                if base_buf not in (None, self):
                    return base_buf.move_inplace(target=target)
                host_mem = memoryview(bytearray(self.size))
                ptr = self._ptr
                rmm._lib.device_buffer.copy_ptr_to_host(self._ptr, host_mem)
                self._ptr_desc["memoryview"] = host_mem
                self._ptr = None
                self._owner = rmm.DeviceBuffer()
                for viewer in self._viewers:
                    viewer._ptr_desc["memoryview"] = host_mem
                    viewer._ptr_desc["offset"] = (viewer._ptr - ptr)
                    viewer._ptr_desc["type"] =  "cpu"
                    viewer._ptr = None
            elif (ptr_type, target) == ("cpu", "gpu"):
                base_buf = get_base_buffer(self)
                if base_buf not in (None, self):
                    return base_buf.move_inplace(target=target)                
                dev_mem = rmm.DeviceBuffer.to_device(
                    self._ptr_desc.pop("memoryview")
                )
                self._ptr = dev_mem.ptr
                self._size = dev_mem.size
                self._owner = dev_mem
                for viewer in self._viewers:
                    viewer._ptr_desc.pop("memoryview", None)
                    offset = viewer._ptr_desc.pop("offset")
                    viewer._ptr = self._ptr + offset
                    viewer._ptr_desc["type"] = "gpu"
            else:
                # TODO: support moving to disk
                raise ValueError(f"Unknown target: {target}")
            self._ptr_desc["type"] = target

    @property
    def ptr(self) -> int:
        if self._spill_manager is not None:
            self._spill_manager.spill_to_device_limit()
        with self._lock:
            self.move_inplace(target="gpu")
            self._ptr_exposed = True
            self._last_accessed = time.monotonic()
            assert self._ptr is not None
            return self._ptr

    @property
    def ptr_exposed(self) -> bool:
        return self._ptr_exposed

    @property
    def spillable(self) -> bool:
        # the viewers should not be "spillable" at all,
        # but it's convenient to be able to invoke the
        # .spillable() method on them.
        return (
            not self._ptr_exposed
            and self._access_counter.use_count() == 1
            and all(v.spillable for v in self._viewers)
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size

    @property
    def last_accessed(self) -> float:
        return self._last_accessed

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def to_host_array(self):
        data = np.empty((self.size,), "u1")
        self.move_inplace(target="gpu")
        rmm._lib.device_buffer.copy_ptr_to_host(self._ptr, data)
        return data

    def _init_from_array_like(self, data, owner):

        if hasattr(data, "__cuda_array_interface__"):
            confirm_1d_contiguous(data.__cuda_array_interface__)
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            self._ptr = ptr
            self._size = size
            self._owner = owner or data
        elif hasattr(data, "__array_interface__"):
            confirm_1d_contiguous(data.__array_interface__)
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = rmm.DeviceBuffer(ptr=ptr, size=size)
            self._init_from_array_like(dbuf, owner)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )

    def serialize(self) -> Tuple[dict, list]:
        ptr_exposed = self._ptr_exposed
        header = {}  # type: Dict[Any, Any]
        header["type-serialized"] = pickle.dumps(type(self))
        header["constructor-kwargs"] = {}
        header["desc"] = self.__cuda_array_interface__.copy()
        header["desc"]["strides"] = (1,)
        header["frame_count"] = 1
        frames = [self]
        self._ptr_exposed = ptr_exposed
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Buffer:
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize Buffer with a single frame."
        func = Buffer.from_buffer if isinstance(frames[0], cls) else cls
        buf = func(frames[0], **header["constructor-kwargs"])
        ptr_exposed = buf._ptr_exposed
        if header["desc"]["shape"] != buf.__cuda_array_interface__["shape"]:
            raise ValueError(
                f"Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {buf.__cuda_array_interface__['shape']}"
            )
        buf._ptr_exposed = ptr_exposed
        return buf

    @classmethod
    def empty(cls, size: int) -> Buffer:
        return Buffer(rmm.DeviceBuffer(size=size))

    def copy(self) -> Buffer:
        """
        Create a new Buffer containing a copy of the data contained
        in this Buffer.
        """
        from rmm._lib.device_buffer import copy_device_to_ptr

        out = Buffer.empty(size=self.size)
        copy_device_to_ptr(self._ptr, out.ptr, self.size)
        return out

    def __repr__(self) -> str:
        if self._ptr is None:
            data_info = str(self._ptr_desc)
        else:
            data_info = str(hex(self._ptr))
        return (
            f"<cudf.core.buffer.Buffer size={format_bytes(self._size)} "
            f"spillable={self.spillable} "
            f"ptr_exposed={self.ptr_exposed} "
            f"access_counter={self._access_counter.use_count()} "
            f"ptr={data_info} owner={repr(self._owner)}>\n"
        )


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    if ptr is None:
        ptr = 0
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    shape = (
        array_interface["shape"] if len(array_interface["shape"]) > 0 else (1,)
    )
    size = functools.reduce(operator.mul, shape)
    return ptr, size * itemsize


def confirm_1d_contiguous(array_interface):
    strides = array_interface["strides"]
    shape = array_interface["shape"]
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    typestr = array_interface["typestr"]
    if typestr not in ("|i1", "|u1"):
        raise TypeError("Buffer data must be of uint8 type")
    if not get_c_contiguity(shape, strides, itemsize):
        raise ValueError("Buffer data must be 1D C-contiguous")


def get_c_contiguity(shape, strides, itemsize):
    """
    Determine if combination of array parameters represents a
    c-contiguous array.
    """
    ndim = len(shape)
    assert strides is None or ndim == len(strides)

    if ndim == 0 or strides is None or (ndim == 1 and strides[0] == itemsize):
        return True

    # any dimension zero, trivial case
    for dim in shape:
        if dim == 0:
            return True

    for this_dim, this_stride in zip(shape, strides):
        if this_stride != this_dim * itemsize:
            return False
    return True
