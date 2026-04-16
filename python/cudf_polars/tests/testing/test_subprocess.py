# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cudf_polars.testing.subprocess utilities."""

from __future__ import annotations

import multiprocessing
import time
from queue import Empty

import pytest

from cudf_polars.testing.subprocess import run_in_subprocess, terminate_process

# -- helpers used as Process targets ------------------------------------------


def _succeed():
    pass


def _succeed_with_result(result_queue):
    result_queue.put("done")


def _raise_runtime_error():
    raise RuntimeError("subprocess raised")


def _hang_forever():
    time.sleep(3600)


# -- run_in_subprocess --------------------------------------------------------


@pytest.mark.parametrize("mp_context", ["fork", "spawn"])
def test_run_in_subprocess_success(mp_context):
    """Target runs without error — result arrives, error queue stays empty."""
    ctx = multiprocessing.get_context(mp_context)
    result_q = ctx.Queue()
    error_q = ctx.Queue()
    proc = ctx.Process(
        target=run_in_subprocess,
        args=(_succeed_with_result, error_q, result_q),
    )
    proc.start()
    proc.join()

    assert proc.exitcode == 0
    assert result_q.get_nowait() == "done"
    with pytest.raises(Empty):
        error_q.get_nowait()
    proc.close()


@pytest.mark.parametrize("mp_context", ["fork", "spawn"])
def test_run_in_subprocess_failure(mp_context):
    """Target raises — exit code is non-zero and traceback is in queue."""
    ctx = multiprocessing.get_context(mp_context)
    error_q = ctx.Queue()
    proc = ctx.Process(
        target=run_in_subprocess,
        args=(_raise_runtime_error, error_q),
    )
    proc.start()
    proc.join()

    assert proc.exitcode != 0
    tb = error_q.get_nowait()
    assert "RuntimeError" in tb
    assert "subprocess raised" in tb
    proc.close()


# -- terminate_process --------------------------------------------------------


def test_terminate_process_clean_exit():
    """Process exits with code 0 — no error raised."""
    proc = multiprocessing.Process(target=_succeed)
    proc.start()
    terminate_process(proc)


def test_terminate_process_nonzero_exit_generic_message():
    """Non-zero exit without a queue raises a generic RuntimeError."""
    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(target=_raise_runtime_error)
    proc.start()
    with pytest.raises(RuntimeError, match="did not exit cleanly"):
        terminate_process(proc)


def test_terminate_process_nonzero_exit_with_traceback():
    """Non-zero exit with error queue includes the child traceback."""
    ctx = multiprocessing.get_context("fork")
    error_q = ctx.Queue()
    proc = ctx.Process(
        target=run_in_subprocess,
        args=(_raise_runtime_error, error_q),
    )
    proc.start()
    with pytest.raises(RuntimeError, match=r"Subprocess traceback:"):
        terminate_process(proc, error_queue=error_q)


def test_terminate_process_traceback_contains_cause():
    """The surfaced traceback includes the original exception details."""
    ctx = multiprocessing.get_context("fork")
    error_q = ctx.Queue()
    proc = ctx.Process(
        target=run_in_subprocess,
        args=(_raise_runtime_error, error_q),
    )
    proc.start()
    proc.join()
    assert proc.exitcode != 0

    tb = error_q.get_nowait()
    assert "RuntimeError" in tb
    assert "subprocess raised" in tb
    proc.close()


def test_terminate_process_timeout_kills_process():
    """Hung process is killed after timeout."""
    proc = multiprocessing.Process(target=_hang_forever)
    proc.start()
    with pytest.raises(RuntimeError, match="did not exit within"):
        terminate_process(proc, timeout=0.5)
