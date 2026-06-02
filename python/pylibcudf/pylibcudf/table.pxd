# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef class Table:
    # List[pylibcudf.Column]
    cdef public list _columns
    # Optional explicit row count, used to preserve the row count of a table
    # with zero columns (which otherwise reports zero rows). None means "derive
    # from the columns". See https://github.com/rapidsai/cudf/issues/21428
    cdef public object _num_rows

    cdef table_view view(self) nogil

    cpdef int num_columns(self)
    cpdef int num_rows(self)
    cpdef tuple shape(self)

    @staticmethod
    cdef Table from_libcudf(
        unique_ptr[table] libcudf_tbl,
        object stream,
        DeviceMemoryResource mr
    )

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner)

    @staticmethod
    cdef Table from_table_view_of_arbitrary(
        const table_view& tv,
        object owner,
        object stream,
    )

    cpdef list columns(self)
    cpdef Table copy(self, object stream = *, DeviceMemoryResource mr=*)
