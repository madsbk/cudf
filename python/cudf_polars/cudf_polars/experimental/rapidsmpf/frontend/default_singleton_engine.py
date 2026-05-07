# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU, single-instance specialization of :class:`SPMDEngine`."""

from __future__ import annotations

import atexit
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar

from rapidsmpf.communicator.single import (
    new_communicator as single_communicator,
)
from rapidsmpf.progress_thread import ProgressThread

from cudf_polars.experimental.rapidsmpf.frontend.core import (
    resolve_rapidsmpf_options,
)
from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

if TYPE_CHECKING:
    from concurrent.futures import Future

__all__ = ["DefaultSingletonEngine", "check_no_live_default_singleton"]


def check_no_live_default_singleton(self_engine: object) -> None:
    """
    Raise if the default singleton engine is alive.

    Parameters
    ----------
    self_engine
        The engine instance being constructed.

    Raises
    ------
    RuntimeError
        If a :class:`DefaultSingletonEngine` is currently alive and
        ``self_engine`` is not itself a :class:`DefaultSingletonEngine`.
    """
    if isinstance(self_engine, DefaultSingletonEngine):
        return
    if isinstance(self_engine, DefaultSingletonEngine):
        return
    with DefaultSingletonEngine._lock:
        if DefaultSingletonEngine._live_instance is not None:
            raise RuntimeError(
                f"Cannot construct {type(self_engine).__name__} while the "
                'default GPU engine (e.g. `.collect(engine="gpu")`) is '
                "active. While the default engine is in use, no explicit "
                "streaming engines may exist. Shut down the default engine "
                "first."
            )


class DefaultSingletonEngine(SPMDEngine):
    """
    Process-wide single-GPU singleton specialization of :class:`SPMDEngine`.

    Always builds a single-rank communicator. At most one live
    instance may exist per process; calling
    ``DefaultSingletonEngine()`` is a get-or-create operation that
    returns the existing instance when available.

    The constructor takes no arguments and always uses default
    RapidsMPF, executor, and engine settings from the environment.
    Users needing custom configuration should construct an
    :class:`SPMDEngine` directly instead.

    Raises
    ------
    RuntimeError
        If another :class:`StreamingEngine` is already alive when
        constructing the singleton.

    Examples
    --------
    >>> with DefaultSingletonEngine() as engine:  # doctest: +SKIP
    ...     result = df.lazy().collect(engine=engine)
    """

    _live_instance: ClassVar[DefaultSingletonEngine | None] = None
    _atexit_registered: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _executor: ClassVar[ThreadPoolExecutor | None] = None
    _creation_future: ClassVar[Future[DefaultSingletonEngine] | None] = None
    SHUTDOWN_TIMEOUT_SECONDS: ClassVar[float] = 10.0

    def __new__(cls) -> DefaultSingletonEngine:
        """
        Return the live singleton, constructing one if needed.

        Raises
        ------
        RuntimeError
            On the cold path, if any other :class:`StreamingEngine` is
            currently alive.
        """
        with cls._lock:
            if cls._live_instance is not None:
                return cls._live_instance
            if cls._executor is None:
                # Cold path: pre-flight active-engine check on the calling
                # thread so the error reports a clean stack rather than one
                # buried inside the worker.
                active_count = SPMDEngine._active_engine_count()
                if active_count > 0:
                    raise RuntimeError(
                        f"Cannot start the default GPU engine (e.g. "
                        f'`.collect(engine="gpu")`) while {active_count} '
                        "explicit streaming engine(s) are alive. While "
                        "explicit engines are in use, the default engine "
                        "cannot also exist. Shut them down first or exit "
                        "their `with` blocks."
                    )
                cls._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="default-singleton-engine",
                )
                cls._creation_future = cls._executor.submit(cls._construct_on_worker)
            executor = cls._executor
            creation_future = cls._creation_future
        assert creation_future is not None
        try:
            instance = creation_future.result()
        except BaseException:
            # Construction failed; reset state so a retry can spawn a fresh
            # executor. Only clear if no other shutdown/retry has already
            # taken over.
            with cls._lock:
                if cls._executor is executor:
                    cls._executor.shutdown(wait=False)
                    cls._executor = None
                    cls._creation_future = None
            raise
        with cls._lock:
            if not cls._atexit_registered:
                atexit.register(cls._atexit_shutdown)
                cls._atexit_registered = True
        return instance

    def __init__(self) -> None:
        # No-op to prevent delegating to ``SPMDEngine.__init__``.
        return

    @classmethod
    def _construct_on_worker(cls) -> DefaultSingletonEngine:
        """
        Build a fully-initialized instance on the dedicated worker thread.

        Bypasses ``cls.__new__`` (which would itself try to dispatch to the
        worker) and runs the parent ``SPMDEngine.__init__`` directly so the
        rapidsmpf Context is constructed on this (the worker) thread.
        """
        instance = object.__new__(cls)
        comm = single_communicator(
            progress_thread=ProgressThread(),
            options=resolve_rapidsmpf_options(None),
        )
        SPMDEngine.__init__(instance, comm=comm)
        assert instance.nranks == 1
        with cls._lock:
            cls._live_instance = instance
        return instance

    def shutdown(self) -> None:
        """
        Shut down the engine and free the singleton slot.

        Submits the teardown to the dedicated worker and waits up to
        :attr:`SHUTDOWN_TIMEOUT_SECONDS`. On timeout a :class:`UserWarning` is
        emitted, the singleton slot is cleared so a fresh ``DefaultSingletonEngine()``
        can spawn a new executor, and the executor handle is released via
        ``executor.shutdown(wait=False)``. The stuck worker is not forcibly detached:
        if it never unblocks, Python's interpreter-exit handler will block joining it.

        Idempotent.
        """
        cls = type(self)
        with cls._lock:
            executor = cls._executor

        if executor is None:
            # Already shut down, or the engine never reached worker-managed
            # construction (e.g. a synthetic instance built bypassing
            # ``__new__`` in tests). Mirror the StreamingEngine semantics.
            with cls._lock:
                if cls._live_instance is self:
                    cls._live_instance = None
                if cls._atexit_registered:
                    atexit.unregister(cls._atexit_shutdown)
                    cls._atexit_registered = False
            super().shutdown()
            return

        future = executor.submit(SPMDEngine.shutdown, self)
        try:
            future.result(timeout=cls.SHUTDOWN_TIMEOUT_SECONDS)
            completed = True
        except TimeoutError:
            completed = False

        with cls._lock:
            if cls._atexit_registered:
                atexit.unregister(cls._atexit_shutdown)
                cls._atexit_registered = False
            if cls._executor is executor:
                cls._live_instance = None
                cls._executor = None
                cls._creation_future = None

        # Release our handle to the executor either way. On success the
        # worker is idle and will exit on the queue sentinel; on timeout
        # the worker stays alive until it eventually unblocks (or until
        # interpreter shutdown, which will block joining it).
        executor.shutdown(wait=False)
        if not completed:
            # The leaked instance is intentionally left in
            # ``_active_engines``: that prevents a subsequent
            # ``DefaultSingletonEngine()`` (or any other streaming engine)
            # from spinning up while the rapidsmpf Context of the previous
            # engine is in an indeterminate state.
            warnings.warn(
                f"DefaultSingletonEngine shutdown did not complete within "
                f"{cls.SHUTDOWN_TIMEOUT_SECONDS}s; the worker thread is "
                "leaked and rapidsmpf resources may not have been released. "
                "No new streaming engine can be created in this process "
                "until the leaked worker eventually returns.",
                stacklevel=2,
            )

    @classmethod
    def _atexit_shutdown(cls) -> None:
        with cls._lock:
            inst = cls._live_instance
        if inst is not None:
            inst.shutdown()
