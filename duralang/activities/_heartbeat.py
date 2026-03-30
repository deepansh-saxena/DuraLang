"""Periodic heartbeats during long-running operations."""

from __future__ import annotations

import asyncio

from temporalio import activity


async def with_heartbeats(coro, message: str = "still running", interval: float = 15.0):
    """Wrap a coroutine with periodic Temporal heartbeats.

    Sends a heartbeat every `interval` seconds while the coroutine runs,
    preventing Temporal from marking the activity as stalled.
    """
    task = asyncio.create_task(coro)
    elapsed = 0
    while not task.done():
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            elapsed += interval
            activity.heartbeat(f"{message} ({elapsed:.0f}s)")
    return task.result()
