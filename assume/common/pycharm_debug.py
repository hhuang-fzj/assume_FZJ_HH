# assume/common/pycharm_debug.py

from __future__ import annotations

import asyncio
from typing import Awaitable


_INSTALLED = False


def install_pycharm_tracing_task_factory() -> None:
    """
    Debug-only helper for PyCharm SSH debugging with Mango asyncio tasks.

    Enable with:
        FORCE_PYCHARM_TRACE_TASKS=1

    This installs a task factory on the currently running event loop. Every
    newly created asyncio task is wrapped so pydevd tracing is re-installed
    at the beginning of that task.
    """
    global _INSTALLED

    if _INSTALLED:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    old_factory = loop.get_task_factory()

    async def traced_coro(coro: Awaitable):
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            suspend=False,
            trace_only_current_thread=False,
            overwrite_prev_trace=True,
        )
        return await coro

    def factory(loop, coro, *args, **kwargs):
        wrapped = traced_coro(coro)

        if old_factory is not None:
            return old_factory(loop, wrapped, *args, **kwargs)

        return asyncio.Task(wrapped, loop=loop)

    loop.set_task_factory(factory)
    _INSTALLED = True
    print(">>> Installed PyCharm tracing task factory for asyncio tasks", flush=True)