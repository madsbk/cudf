# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parallel Agg Logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.expr import Agg, Col, NamedExpr
from cudf_polars.dsl.ir import IR, broadcast
from cudf_polars.experimental.base import _concat, get_key_name
from cudf_polars.experimental.dispatch import generate_ir_tasks

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence

    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.typing import Schema


class SelectAgg(IR):
    """Multi-partition aggregation."""

    __slots__ = ("exprs", "should_broadcast")
    _non_child = ("schema", "exprs", "should_broadcast")
    expr: tuple[NamedExpr, ...]
    """Aggregation expressions to evaluate to form the new dataframe."""
    should_broadcast: bool
    """Should columns be broadcast?"""

    def __init__(
        self,
        schema: Schema,
        exprs: Sequence[NamedExpr],
        should_broadcast: bool,  # noqa: FBT001
        df: IR,
    ):
        self.schema = schema
        self.exprs = tuple(exprs)
        self.should_broadcast = should_broadcast
        self.children = (df,)
        self._non_child_args = (self.exprs, should_broadcast)

    @classmethod
    def do_evaluate(
        cls,
        exprs: tuple[NamedExpr, ...],
        should_broadcast: bool,  # noqa: FBT001
        df: DataFrame,
    ) -> DataFrame:
        """Evaluate and return a dataframe."""
        # Handle any broadcasting
        columns = [e.evaluate(df) for e in exprs]
        if should_broadcast:
            columns = broadcast(*columns)
        return DataFrame(columns)


def _evaluate(
    exprs: tuple[NamedExpr],
    should_broadcast: bool,  # noqa: FBT001
    dfs: list[DataFrame],
):
    df = _concat(dfs) if len(dfs) > 1 else dfs[0]
    return SelectAgg.do_evaluate(
        exprs,
        should_broadcast,
        df,
    )


@generate_ir_tasks.register(SelectAgg)
def _(
    ir: SelectAgg, partition_info: MutableMapping[IR, PartitionInfo]
) -> MutableMapping[Any, Any]:
    # Extract aggregation info
    aggs = []
    for named_expr in ir.exprs:
        expr = named_expr.value
        if isinstance(expr, Agg) and expr.name in {"sum"}:
            part_expr = named_expr
            comb_expr = named_expr
            col_name = named_expr.name
            final_expr = NamedExpr(
                col_name,
                Col(ir.schema.get(col_name), col_name),
            )
        else:
            raise NotImplementedError()
        aggs.append(
            {
                "part": part_expr,
                "combine": comb_expr,
                "final": final_expr,
            }
        )

    # Build graph
    (df,) = ir.children
    should_broadcast = ir.should_broadcast
    name = get_key_name(ir)
    part_name = f"part-{name}"
    child_name = get_key_name(df)
    child_count = partition_info[df].count
    graph: MutableMapping[Any, Any] = {
        (part_name, part_id): (
            _evaluate,
            tuple(agg["part"] for agg in aggs),
            should_broadcast,
            [(child_name, part_id)],
        )
        for part_id in range(child_count)
    }
    graph[(name, 0)] = (
        _evaluate,
        tuple(agg["final"] for agg in aggs),
        should_broadcast,
        [
            (
                _evaluate,
                tuple(agg["combine"] for agg in aggs),
                should_broadcast,
                list(graph.keys()),
            )
        ],
    )
    return graph
