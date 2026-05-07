# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for :class:`DefaultSingletonEngine`.

Every test body runs inside a worker spawned by the module-scoped
``proc_pool`` fixture. This isolates us from the session-scoped
``streaming_engines`` fixture in :file:`conftest.py`, which creates an
``SPMDEngine`` that lives for the entire pytest session and would
otherwise trip the "no other engine alive when default is created"
guardrail in :class:`DefaultSingletonEngine`.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


# ---------------------------------------------------------------------------
# Subprocess infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def proc_pool() -> Generator[ProcessPoolExecutor, None, None]:
    """
    Module-scoped ``ProcessPoolExecutor`` used to run test bodies in isolation.

    Spawn (rather than fork) is used so each worker starts from a clean
    interpreter state — no inherited rapidsmpf Context, no inherited
    pytest fixtures, no inherited GPU resources.
    """
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        yield pool


def _reset_singleton_class_state() -> None:
    """Tear down any leftover engine and reset every class-level slot."""
    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    if DefaultSingletonEngine._live_instance is not None:
        DefaultSingletonEngine._live_instance.shutdown()
    DefaultSingletonEngine._atexit_registered = False
    if DefaultSingletonEngine._executor is not None:
        DefaultSingletonEngine._executor.shutdown(wait=False)
        DefaultSingletonEngine._executor = None
    DefaultSingletonEngine._creation_future = None


def _run(body: object) -> None:
    """
    Subprocess entry point.

    Resets the singleton class state, runs ``body``, and resets again on
    exit so a stuck test doesn't bleed state into the next worker
    invocation. ``body`` must be a top-level callable so it is picklable
    across the spawn boundary.
    """
    _reset_singleton_class_state()
    try:
        body()  # type: ignore[operator]
    finally:
        _reset_singleton_class_state()


# ---------------------------------------------------------------------------
# Test bodies
# ---------------------------------------------------------------------------


def _body_lifecycle() -> None:
    """
    Construction, type, get-or-create, shutdown, context manager, fresh-after-shutdown.
    """
    import pytest

    import polars as pl

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    # Construction succeeds; isinstance check skips its parent's
    # ``check_no_live_default_singleton`` so SPMDEngine.__init__ runs cleanly.
    with DefaultSingletonEngine() as engine:
        assert engine.nranks == 1
        assert engine.rank == 0
        assert isinstance(engine, SPMDEngine)
        assert isinstance(engine, pl.GPUEngine)
        # Get-or-create: a second call returns the same instance.
        assert DefaultSingletonEngine() is engine

    # After context-manager exit, the slot is free.
    e2 = DefaultSingletonEngine()
    assert e2 is not engine
    e2.shutdown()
    e2.shutdown()  # idempotent

    # Comm/context properties raise after shutdown.
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = e2.comm
    with pytest.raises(RuntimeError, match="shutdown"):
        _ = e2.context

    # Constructor takes no kwargs.
    with pytest.raises(TypeError):
        DefaultSingletonEngine(comm=object())  # type: ignore[call-arg]


def test_lifecycle(proc_pool: ProcessPoolExecutor) -> None:
    """Basic construction, get-or-create, shutdown, and constructor-surface checks."""
    proc_pool.submit(_run, _body_lifecycle).result()


def _body_default_path_routing() -> None:
    """
    .collect(engine=pl.GPUEngine("streaming")) routes through the singleton,
    reuses it across queries, and picks up an explicit user singleton.
    """
    import polars as pl

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    # Cold path: vanilla streaming GPUEngine triggers DefaultSingletonEngine.
    assert DefaultSingletonEngine._live_instance is None
    engine = pl.GPUEngine(executor="streaming")
    result = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine=engine)
    assert result.shape == (3, 2)
    assert result["a"].to_list() == [1, 2, 3]

    # Second query reuses the same singleton.
    first = DefaultSingletonEngine._live_instance
    assert first is not None
    pl.LazyFrame({"a": [4, 5, 6]}).collect(engine=engine)
    assert DefaultSingletonEngine._live_instance is first
    first.shutdown()

    # An already-live user singleton is reused by the default path.
    with DefaultSingletonEngine() as user_engine:
        pl.LazyFrame({"a": [1]}).collect(engine=engine)
        assert DefaultSingletonEngine._live_instance is user_engine


def test_default_path_routing(proc_pool: ProcessPoolExecutor) -> None:
    """``cluster='single'`` routes through the singleton end-to-end."""
    proc_pool.submit(_run, _body_default_path_routing).result()


def _body_concurrent_warm_path() -> None:
    """Concurrent ``DefaultSingletonEngine()`` calls return the same instance."""
    import threading

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    main_engine = DefaultSingletonEngine()
    try:
        barrier = threading.Barrier(8)
        results: list[DefaultSingletonEngine] = []
        results_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            with results_lock:
                results.append(DefaultSingletonEngine())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        assert all(r is main_engine for r in results)
    finally:
        main_engine.shutdown()


def test_concurrent_warm_path(proc_pool: ProcessPoolExecutor) -> None:
    """Concurrent ``DefaultSingletonEngine()`` calls all return the same instance."""
    proc_pool.submit(_run, _body_concurrent_warm_path).result()


def _body_atexit_lifecycle() -> None:
    """
    atexit hook: registered once on first construction, unregistered on
    shutdown, re-registered on a fresh construction after shutdown, and
    registered regardless of which thread first constructs the engine.
    """
    import threading
    from unittest.mock import patch

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )

    # Register-once + unregister-on-shutdown + re-register after shutdown.
    with (
        patch(
            "cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine.atexit.register"
        ) as mock_register,
        patch(
            "cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine.atexit.unregister"
        ) as mock_unregister,
    ):
        e1 = DefaultSingletonEngine()
        DefaultSingletonEngine()  # warm path: no extra register
        DefaultSingletonEngine()  # warm path: no extra register
        assert mock_register.call_count == 1
        assert DefaultSingletonEngine._atexit_registered is True

        e1.shutdown()
        assert mock_unregister.call_count == 1
        assert DefaultSingletonEngine._atexit_registered is False

        e2 = DefaultSingletonEngine()
        try:
            assert mock_register.call_count == 2
        finally:
            e2.shutdown()

    # Off-main construction also registers atexit (delegation through the
    # worker thread makes that safe).
    state: dict[str, bool] = {}
    create_done = threading.Event()

    def thread_target() -> None:
        DefaultSingletonEngine()
        state["registered"] = DefaultSingletonEngine._atexit_registered
        create_done.set()

    t = threading.Thread(target=thread_target)
    t.start()
    create_done.wait()
    t.join()
    try:
        assert state["registered"] is True
    finally:
        live = DefaultSingletonEngine._live_instance
        assert live is not None
        live.shutdown()


def test_atexit_lifecycle(proc_pool: ProcessPoolExecutor) -> None:
    """atexit register / unregister flow over the engine lifetime."""
    proc_pool.submit(_run, _body_atexit_lifecycle).result()


def _body_explicit_engines_blocked_when_singleton_alive() -> None:
    """SPMDEngine, RayEngine, and DaskEngine all refuse to coexist with the singleton."""
    from unittest.mock import patch

    import pytest

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    # SPMDEngine is always available when the rapidsmpf frontend is imported.
    with patch.object(DefaultSingletonEngine, "_live_instance", object()):
        with pytest.raises(RuntimeError, match="default GPU engine"):
            SPMDEngine()

        # Ray (optional dep).
        try:
            from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine
        except ImportError:
            pass
        else:
            with pytest.raises(RuntimeError, match="default GPU engine"):
                RayEngine()

        # Dask (optional dep).
        try:
            from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine
        except ImportError:
            pass
        else:
            with pytest.raises(RuntimeError, match="default GPU engine"):
                DaskEngine()


def test_explicit_engines_blocked_when_singleton_alive(
    proc_pool: ProcessPoolExecutor,
) -> None:
    """SPMD/Ray/Dask refuse to coexist with a live ``DefaultSingletonEngine``."""
    proc_pool.submit(_run, _body_explicit_engines_blocked_when_singleton_alive).result()


def _body_singleton_blocked_when_explicit_alive() -> None:
    """
    The reverse direction: ``DefaultSingletonEngine()`` refuses if any
    other ``StreamingEngine`` is alive. Verifies both the synthetic
    sentinel and a real SPMDEngine path, plus the
    ``_active_engine_count`` accessor.
    """
    import weakref
    from unittest.mock import patch

    import pytest

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    assert StreamingEngine._active_engine_count() == 0

    # Synthetic sentinel — fast, no real engine construction.
    class _FakeEngine:
        pass

    fake = _FakeEngine()
    with (
        patch.object(StreamingEngine, "_active_engines", weakref.WeakSet({fake})),
        pytest.raises(RuntimeError, match="explicit streaming engine"),
    ):
        DefaultSingletonEngine()

    # A real explicit SPMDEngine also blocks the default.
    with (
        SPMDEngine(),
        pytest.raises(RuntimeError, match="explicit streaming engine"),
    ):
        DefaultSingletonEngine()
    assert StreamingEngine._active_engine_count() == 0

    # Active-count tracks DefaultSingletonEngine's own lifecycle too.
    with DefaultSingletonEngine():
        assert StreamingEngine._active_engine_count() == 1
    assert StreamingEngine._active_engine_count() == 0


def test_singleton_blocked_when_explicit_alive(
    proc_pool: ProcessPoolExecutor,
) -> None:
    """``DefaultSingletonEngine()`` refuses if any other StreamingEngine is alive."""
    proc_pool.submit(_run, _body_singleton_blocked_when_explicit_alive).result()


def _body_worker_thread_isolation() -> None:
    """
    The dedicated worker thread owns construction and shutdown.

    - Construction runs on the named worker thread, not the caller's.
    - ``shutdown`` from a non-creator thread (i.e. the test main thread)
      doesn't crash, because the teardown is dispatched back to the
      worker.
    - If construction raises on the worker, the caller sees the same
      exception and the slot is reset for retry.
    """
    import threading
    from typing import Any
    from unittest.mock import patch

    import pytest

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    # 1) Construction runs on the worker thread.
    recorded: dict[str, threading.Thread] = {}
    real_init = SPMDEngine.__init__

    def recording_init(self: SPMDEngine, **kwargs: Any) -> None:
        recorded["thread"] = threading.current_thread()
        real_init(self, **kwargs)

    with patch.object(SPMDEngine, "__init__", recording_init):
        engine = DefaultSingletonEngine()
    try:
        assert recorded["thread"] is not threading.current_thread()
        assert recorded["thread"].name.startswith("default-singleton-engine")
    finally:
        engine.shutdown()

    # 2) Cross-thread shutdown via the worker is safe.
    create_done = threading.Event()

    def creator() -> None:
        DefaultSingletonEngine()
        create_done.set()

    t = threading.Thread(target=creator)
    t.start()
    create_done.wait()
    t.join()
    live = DefaultSingletonEngine._live_instance
    assert live is not None
    live.shutdown()  # different thread than the one that constructed
    assert DefaultSingletonEngine._live_instance is None
    assert DefaultSingletonEngine._executor is None

    # 3) Construction error propagates and resets state for a retry.
    def broken_init(self: SPMDEngine, **kwargs: object) -> None:
        raise RuntimeError("synthetic boom")

    with (
        patch.object(SPMDEngine, "__init__", broken_init),
        pytest.raises(RuntimeError, match="synthetic boom"),
    ):
        DefaultSingletonEngine()
    assert DefaultSingletonEngine._live_instance is None
    assert DefaultSingletonEngine._executor is None
    DefaultSingletonEngine().shutdown()  # retry succeeds


def test_worker_thread_isolation(proc_pool: ProcessPoolExecutor) -> None:
    """Worker-thread ownership of the engine lifecycle."""
    proc_pool.submit(_run, _body_worker_thread_isolation).result()


def _body_shutdown_timeout() -> None:
    """
    A hung ``SPMDEngine.shutdown`` causes the timeout branch to fire:
    a warning is emitted, the singleton slot is cleared, and any new
    construction is refused until the leaked worker eventually returns.
    """
    import threading
    from unittest.mock import patch

    import pytest

    from cudf_polars.experimental.rapidsmpf.frontend.default_singleton_engine import (
        DefaultSingletonEngine,
    )
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    release_worker = threading.Event()
    real_done = threading.Event()
    real_shutdown = SPMDEngine.shutdown

    def hanging_super_shutdown(self: SPMDEngine) -> None:
        release_worker.wait()
        try:
            # Run the real teardown so the rapidsmpf Context is destroyed
            # on the construction (worker) thread, otherwise GC of the
            # engine on the wrong thread crashes with
            # "Context::shutdown() called from a different thread...".
            real_shutdown(self)
        finally:
            real_done.set()

    with patch.object(DefaultSingletonEngine, "SHUTDOWN_TIMEOUT_SECONDS", 0.1):
        engine = DefaultSingletonEngine()
        with patch.object(SPMDEngine, "shutdown", hanging_super_shutdown):
            with pytest.warns(UserWarning, match="did not complete within"):
                engine.shutdown()
            # Slot cleared, but the leaked engine is still in the
            # active-engine registry — no new construction allowed.
            assert DefaultSingletonEngine._live_instance is None
            assert DefaultSingletonEngine._executor is None
            with pytest.raises(RuntimeError, match="explicit streaming engine"):
                DefaultSingletonEngine()
            # Once the leaked worker returns it removes itself from the
            # registry; a fresh construction is allowed again.
            release_worker.set()
            real_done.wait()
            DefaultSingletonEngine().shutdown()


def test_shutdown_timeout(proc_pool: ProcessPoolExecutor) -> None:
    """Hung shutdown emits a warning and refuses new construction until unblock."""
    proc_pool.submit(_run, _body_shutdown_timeout).result()
