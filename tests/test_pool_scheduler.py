from __future__ import annotations

import threading
import time

from blackbase.resources import PoolScheduler


def test_executor_view_reserves_and_releases_capacity() -> None:
    pool = PoolScheduler(total_threads=4)
    try:
        with pool.as_executor(3) as executor:
            assert pool.available() == 1
            assert list(executor.map(lambda value: value * 2, range(4))) == [0, 2, 4, 6]
        assert pool.available() == 4
    finally:
        pool.shutdown(wait=True)


def test_executor_view_releases_after_nonblocking_shutdown_finishes_tasks() -> None:
    pool = PoolScheduler(total_threads=1)
    entered = threading.Event()
    release = threading.Event()

    def blocking() -> str:
        entered.set()
        release.wait(timeout=2.0)
        return "done"

    view = pool.as_executor(1)
    executor = view.__enter__()
    future = executor.submit(blocking)
    assert entered.wait(timeout=1.0)

    view.shutdown(wait=False)
    assert pool.available() == 0
    release.set()
    assert future.result(timeout=1.0) == "done"

    deadline = time.monotonic() + 1.0
    while pool.available() != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.available() == 1
    pool.shutdown(wait=True)
