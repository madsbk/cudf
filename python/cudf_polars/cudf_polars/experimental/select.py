# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Select Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Agg
from cudf_polars.dsl.ir import Select
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.dispatch import lower_ir_node

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import LowerIRTransformer


@lower_ir_node.register(Select)
def _(
    ir: Select, rec: LowerIRTransformer
) -> tuple[IR, MutableMapping[IR, PartitionInfo]]:
    child, partition_info = rec(ir.children[0])
    pi = partition_info[child]
    new_node: SelectAgg | Select

    # Check if we are directly selecting an Agg.
    # To handle arbitrary Agg expressions, we will need
    # a mechanism to decompose an Expr graph containing
    # 1+ nodes that are non pointwise.
    if pi.count > 1 and all(isinstance(expr.value, Agg) for expr in ir.exprs):
        from cudf_polars.experimental.agg import SelectAgg

        new_node = SelectAgg(
            ir.schema,
            ir.exprs,
            ir.should_broadcast,
            child,
        )
        partition_info[new_node] = PartitionInfo(count=1)
        return new_node, partition_info

    elif pi.count > 1 and not all(
        expr.is_pointwise for expr in traversal([e.value for e in ir.exprs])
    ):
        # TODO: Handle non-pointwise expressions.
        raise NotImplementedError(
            f"Selection {ir} does not support multiple partitions."
        )
    new_node = ir.reconstruct([child])
    partition_info[new_node] = pi
    return new_node, partition_info
