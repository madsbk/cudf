# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess testing utilities for cudf_polars."""

from __future__ import annotations

import queue as _queue_module
import traceback
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.queues import Queue as MPQueue

__all__: list[str] = ["run_in_subprocess", "terminate_process"]


def run_in_subprocess(
    target: Any,
    error_queue: MPQueue[str],
    *args: Any,
) -> None:
    """
    Run *target* in a subprocess, forwarding exceptions to *error_queue*.

    Use this as the ``target`` for :class:`multiprocessing.Process`,
    passing the same *error_queue* to :func:`terminate_process`.  Any
    unhandled exception raised by *target* is serialised as a formatted
    traceback string and placed in *error_queue* before being re-raised,
    so the process still exits with a non-zero code.

    Parameters
    ----------
    target
        The callable to execute in the child process.
    error_queue
        A :class:`multiprocessing.Queue` into which the formatted
        traceback string is placed on failure.
    *args
        Positional arguments forwarded to *target*.
    """
    try:
        target(*args)
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def terminate_process(
    proc: Process,
    *,
    timeout: float = 30,
    error_queue: MPQueue[str] | None = None,
) -> None:
    """
    Join *proc* and raise :class:`RuntimeError` on failure.

    Parameters
    ----------
    proc
        The child process to wait for.
    timeout
        Seconds to wait before force-killing the process.
    error_queue
        When the subprocess was started via :func:`run_in_subprocess`,
        pass the same queue here.  If the process exited with a non-zero
        code, ``terminate_process`` will pull the formatted traceback
        from the queue and include it in the raised :class:`RuntimeError`,
        making the root cause visible without digging into subprocess
        logs.
    """
    try:
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()
            proc.join()
            raise RuntimeError(f"Process {proc.pid} did not exit within {timeout}s")
        if proc.exitcode != 0:
            msg = f"Process did not exit cleanly (exit code: {proc.exitcode})"
            if error_queue is not None:
                try:
                    tb = error_queue.get_nowait()
                    raise RuntimeError(f"{msg}\n\nSubprocess traceback:\n{tb}")
                except _queue_module.Empty:
                    pass
            raise RuntimeError(msg)
    finally:
        proc.close()
