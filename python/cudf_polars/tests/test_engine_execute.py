# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the engine's execute path."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sized

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

from cudf_polars.engine.partitioned_source import PartitionedSource


def _source_lf() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def _unsupported_lf() -> pl.LazyFrame:
    """A query whose translation records an error (unsupported on the GPU)."""
    return (
        pl.LazyFrame({"orderby": [1, 4, 8, 10], "values": [1, 2, 3, 4]})
        .rolling("orderby", period="4i")
        .agg(pl.col("values"))
    )


# ---------------------------------------------------------------------------
# PartitionedSource - host-only, no engine fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [4, None])
def test_partitioned_source_chunks(chunk):
    """A partition is sliced into <= chunk-row SizedChunks whose union is the input."""
    df = pl.DataFrame({"a": list(range(10))})
    src = PartitionedSource({0: lambda: df, 1: lambda: df}, df.schema, chunk)
    out = src(None, None, None, None, rank=1, nranks=2)
    assert isinstance(out, Sized)
    chunks = list(out)
    expected = math.ceil(10 / chunk) if chunk else 1
    assert len(out) == expected == len(chunks)
    assert_frame_equal(pl.concat(chunks), df)


def test_partitioned_source_absent_rank_is_empty():
    """A rank with no partition emits exactly one empty same-schema chunk."""
    df = pl.DataFrame({"a": list(range(10))})
    src = PartitionedSource({0: lambda: df}, df.schema, 4)
    chunks = list(src(None, None, None, None, rank=2, nranks=4))
    assert len(chunks) == 1
    assert chunks[0].height == 0
    assert chunks[0].columns == ["a"]


def test_partitioned_source_projection_and_predicate():
    """Projection and predicate are applied per chunk."""
    df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
    src = PartitionedSource({0: lambda: df}, df.schema, 4)
    out = pl.concat(list(src(["a"], pl.col("a") > 6, None, None, rank=0, nranks=1)))
    assert out.columns == ["a"]
    assert out["a"].to_list() == [7, 8, 9]


# ---------------------------------------------------------------------------
# SPMDQueryResult
# ---------------------------------------------------------------------------


def test_spmd_execute_lazy_roundtrip(spmd_engine):
    """SPMDEngine.execute().lazy() round-trips through the SPMD engine."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    collected = result.lazy().collect(engine=spmd_engine)
    assert_frame_equal(collected, lf.collect(), check_row_order=False)


def test_spmd_execute_lazy_filter(spmd_engine):
    """SPMDEngine.execute().lazy() supports chained filter operations."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    collected = result.lazy().filter(pl.col("a") > 1).collect(engine=spmd_engine)
    assert_frame_equal(
        collected, lf.filter(pl.col("a") > 1).collect(), check_row_order=False
    )


def test_spmd_execute_lazy_projection(spmd_engine):
    """SPMDEngine.execute().lazy() supports projection pushdown."""
    lf = _source_lf()
    result = spmd_engine.execute(lf)
    collected = result.lazy().select("a").collect(engine=spmd_engine)
    assert_frame_equal(collected, lf.select("a").collect(), check_row_order=False)


def test_spmd_execute_unsupported_raises(spmd_engine):
    """execute() rejects a query with a translation error before dispatching it."""
    with pytest.raises(NotImplementedError):
        spmd_engine.execute(_unsupported_lf())


def test_spmd_execute_reset_invalidates_result(spmd_engine):
    """_reset() frees a live result's partition before teardown; reuse raises."""
    result = spmd_engine.execute(_source_lf())
    spmd_engine._reset()
    with pytest.raises(RuntimeError, match="invalidated"):
        result.lazy()
    # The engine is still usable after reset.
    assert spmd_engine.execute(_source_lf()).n_rows_total is not None


# ---------------------------------------------------------------------------
# DaskQueryResult
# ---------------------------------------------------------------------------


def test_dask_execute_lazy_roundtrip(dask_engine):
    """DaskEngine.execute().lazy() round-trips through the Dask engine."""
    lf = _source_lf()
    result = dask_engine.execute(lf)
    collected = result.lazy().collect(engine=dask_engine)
    assert_frame_equal(collected, lf.collect(), check_row_order=False)


def test_dask_execute_lazy_filter(dask_engine):
    """DaskEngine.execute().lazy() supports chained filter operations."""
    lf = _source_lf()
    result = dask_engine.execute(lf)
    collected = result.lazy().filter(pl.col("a") > 1).collect(engine=dask_engine)
    assert_frame_equal(
        collected, lf.filter(pl.col("a") > 1).collect(), check_row_order=False
    )


def test_dask_execute_lazy_projection(dask_engine):
    """DaskEngine.execute().lazy() supports projection pushdown."""
    lf = _source_lf()
    result = dask_engine.execute(lf)
    collected = result.lazy().select("a").collect(engine=dask_engine)
    assert_frame_equal(collected, lf.select("a").collect(), check_row_order=False)


def test_dask_execute_n_partitions(dask_engine):
    """DaskQueryResult tracks one retained partition per Dask worker."""
    result = dask_engine.execute(_source_lf())
    assert len(result._ranks) == dask_engine._nranks


def _stored_count(query_id, *, dask_worker=None):
    """Number of this query's retained partitions in the running process's store."""
    # Imported here, not at module top: this is shipped to the worker/actor
    # process (Dask client.run / Ray _run) and must read *that* process's store.
    # A top-level import would let cloudpickle capture the client's (empty) dict
    # by value. The dask_worker kwarg is injected by client.run and ignored.
    from cudf_polars.engine import dataframe_store

    return sum(1 for k in dataframe_store._STORE if k[0] == query_id)


def _install_store_partition_fault(fail_rank):
    """
    Wrap this process's ``store_partition`` to raise on ``fail_rank`` after storing.

    Shipped to the worker/actor process (Dask ``client.run`` / Ray ``_run``) so it
    patches that process's own module. The original is invoked first, so the
    collective completes and the partition is stored on *every* rank before the
    designated rank raises - reproducing "one rank fails after the others finished
    storing".
    """
    from cudf_polars.engine import dataframe_store

    orig = dataframe_store.store_partition

    def wrapper(ctx, comm, py_executor, ir, config_options, query_id):
        rank = orig(ctx, comm, py_executor, ir, config_options, query_id)
        if comm.rank == fail_rank:
            raise RuntimeError("injected worker failure after store_partition")
        return rank

    dataframe_store._test_orig_store_partition = orig
    dataframe_store.store_partition = wrapper


def _restore_store_partition():
    """Undo :func:`_install_store_partition_fault` on this worker (idempotent)."""
    from cudf_polars.engine import dataframe_store

    orig = getattr(dataframe_store, "_test_orig_store_partition", None)
    if orig is not None:
        dataframe_store.store_partition = orig
        del dataframe_store._test_orig_store_partition


def test_dask_execute_release_reclaims_worker_partitions(dask_engine):
    """release() drops the query's worker-retained partitions (leak fix)."""
    result = dask_engine.execute(_source_lf())
    query_id = result._query_id
    client = dask_engine._dask_ctx.client

    before = client.run(_stored_count, query_id)
    assert sum(before.values()) == dask_engine._nranks
    assert query_id in dask_engine._retained_query_ids

    result.release()
    after = client.run(_stored_count, query_id)
    assert sum(after.values()) == 0
    # release() also removes the query from tracking, so a long-lived engine
    # does not accumulate IDs across execute()/release() cycles.
    assert query_id not in dask_engine._retained_query_ids

    result.release()  # idempotent, no error


def test_dask_backend_failed_drop_keeps_tracking():
    """A drop that fails keeps the query tracked for the shutdown backstop (P2)."""
    import types

    from cudf_polars.engine.dask import DaskRetainedBackend

    class _FailingClient:
        def run(self, *args, **kwargs):
            raise RuntimeError("workers unreachable")

    query_id = uuid.uuid4()
    tracked = {query_id}
    ctx = types.SimpleNamespace(client=_FailingClient(), rapidsmpf_id="uid")
    backend = DaskRetainedBackend(ctx, tracked)

    backend.drop_retained(query_id)  # best-effort: swallows the error ...
    assert query_id in tracked  # ... but keeps the ID so the backstop retries


def test_dask_backend_confirmed_drop_removes_tracking():
    """A drop confirmed on all workers removes the query from tracking (P2)."""
    import types

    from cudf_polars.engine.dask import DaskRetainedBackend

    class _OkClient:
        def run(self, *args, **kwargs):
            return {}

    query_id = uuid.uuid4()
    tracked = {query_id}
    ctx = types.SimpleNamespace(client=_OkClient(), rapidsmpf_id="uid")
    backend = DaskRetainedBackend(ctx, tracked)

    backend.drop_retained(query_id)
    assert query_id not in tracked


def test_dask_execute_reset_drops_and_untracks(dask_engine):
    """_reset() drops retained partitions and clears tracking before teardown."""
    result = dask_engine.execute(_source_lf())
    query_id = result._query_id
    client = dask_engine._dask_ctx.client
    assert query_id in dask_engine._retained_query_ids
    assert sum(client.run(_stored_count, query_id).values()) == dask_engine._nranks

    dask_engine._reset()

    assert query_id not in dask_engine._retained_query_ids
    assert sum(client.run(_stored_count, query_id).values()) == 0


def test_dask_execute_default_engine_collect_raises(dask_engine):
    """A retained result is not collectable off the producing engine."""
    result = dask_engine.execute(_source_lf())
    with pytest.raises(Exception):  # noqa: B017 - GPU frame can't cross to the client
        result.lazy().collect()


def test_dask_execute_partial_failure_drops_all_partitions(dask_engine):
    """
    A rank failing after others stored their partition leaves nothing behind.

    Drives the backend directly so the ``query_id`` is known. A fault makes one
    worker raise *after* storing (every rank has stored by then), so a correct
    cleanup path must broadcast a drop for the query to all workers before
    re-raising - otherwise the successful ranks would orphan their GPU partitions.
    """
    from cudf_polars.dsl.translate import Translator
    from cudf_polars.engine.dask import DaskRetainedBackend

    client = dask_engine._dask_ctx.client
    lf = _source_lf()
    translator = Translator(lf._ldf.visit(), dask_engine)
    ir = translator.translate_ir()
    query_id = uuid.uuid4()
    backend = DaskRetainedBackend(
        dask_engine._dask_ctx, dask_engine._retained_query_ids
    )

    client.run(_install_store_partition_fault, dask_engine._nranks - 1)
    try:
        with pytest.raises(Exception):  # noqa: B017 - injected worker failure
            backend.execute_retained(ir, translator.config_options, query_id)
    finally:
        client.run(_restore_store_partition)

    # Every rank stored before the fault, so a working cleanup must have dropped
    # them all (the failure-path broadcast is idempotent across workers).
    counts = client.run(_stored_count, query_id)
    assert sum(counts.values()) == 0


def test_dask_execute_unsupported_raises(dask_engine):
    """execute() rejects a query with a translation error before dispatching it."""
    with pytest.raises(NotImplementedError):
        dask_engine.execute(_unsupported_lf())


# ---------------------------------------------------------------------------
# RayQueryResult
# ---------------------------------------------------------------------------


def test_ray_execute_lazy_roundtrip(ray_engine):
    """RayEngine.execute().lazy() round-trips through the Ray engine."""
    lf = _source_lf()
    result = ray_engine.execute(lf)
    collected = result.lazy().collect(engine=ray_engine)
    assert_frame_equal(collected, lf.collect(), check_row_order=False)


def test_ray_execute_lazy_filter(ray_engine):
    """RayEngine.execute().lazy() supports chained filter operations."""
    lf = _source_lf()
    result = ray_engine.execute(lf)
    collected = result.lazy().filter(pl.col("a") > 1).collect(engine=ray_engine)
    assert_frame_equal(
        collected, lf.filter(pl.col("a") > 1).collect(), check_row_order=False
    )


def test_ray_execute_lazy_projection(ray_engine):
    """RayEngine.execute().lazy() supports projection pushdown."""
    lf = _source_lf()
    result = ray_engine.execute(lf)
    collected = result.lazy().select("a").collect(engine=ray_engine)
    assert_frame_equal(collected, lf.select("a").collect(), check_row_order=False)


def test_ray_execute_n_partitions(ray_engine):
    """RayQueryResult tracks one retained partition per rank."""
    result = ray_engine.execute(_source_lf())
    assert len(result._ranks) == ray_engine._nranks


def test_ray_execute_unsupported_raises(ray_engine):
    """execute() rejects a query with a translation error before dispatching it."""
    with pytest.raises(NotImplementedError):
        ray_engine.execute(_unsupported_lf())


def test_ray_execute_partial_failure_drops_all_partitions(ray_engine):
    """
    An actor failing after others stored their partition leaves nothing behind.

    Ray analogue of the Dask partial-failure test: a fault makes one actor raise
    *after* storing (every rank has stored by then), so a correct cleanup path
    must broadcast a drop for the query to all actors before re-raising -
    otherwise the successful ranks would orphan their GPU partitions.
    """
    import sys

    import ray
    from ray import cloudpickle

    from cudf_polars.dsl.translate import Translator
    from cudf_polars.engine.ray import RayRetainedBackend

    lf = _source_lf()
    translator = Translator(lf._ldf.visit(), ray_engine)
    ir = translator.translate_ir()
    query_id = uuid.uuid4()
    actors = ray_engine.rank_actors
    backend = RayRetainedBackend(actors)

    # The fault helpers live in this test module, which the actor processes can't
    # import (No module named 'tests'). Tell Ray's cloudpickle to serialize them
    # by value instead of by reference so the actors don't need to import them.
    test_module = sys.modules[__name__]
    cloudpickle.register_pickle_by_value(test_module)
    try:
        ray.get(
            [
                a._run.remote(_install_store_partition_fault, ray_engine._nranks - 1)
                for a in actors
            ]
        )
        try:
            with pytest.raises(Exception):  # noqa: B017 - injected actor failure
                backend.execute_retained(ir, translator.config_options, query_id)
        finally:
            ray.get([a._run.remote(_restore_store_partition) for a in actors])

        # Every rank stored before the fault, so a working cleanup must have
        # dropped them all (the failure-path broadcast is idempotent).
        counts = ray.get([a._run.remote(_stored_count, query_id) for a in actors])
    finally:
        cloudpickle.unregister_pickle_by_value(test_module)

    assert sum(counts) == 0
