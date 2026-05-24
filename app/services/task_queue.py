import asyncio
from collections import deque
from typing import Callable, Optional


class TaskQueue:
    """Queue-based task scheduler with concurrency limit."""

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self.queue: deque = deque()
        self.active_tasks = 0
        self.lock = asyncio.Lock()

    async def submit(self, coro):
        """Submit a coroutine to the queue."""
        self.queue.append(coro)
        await self._process_queue()

    async def _process_queue(self):
        """Process tasks from queue with concurrency limit."""
        async with self.lock:
            while self.queue and self.active_tasks < self.max_concurrent:
                coro = self.queue.popleft()
                self.active_tasks += 1
                asyncio.create_task(self._run_with_cleanup(coro))

    async def _run_with_cleanup(self, coro):
        """Run a task and decrement active counter."""
        try:
            await coro
        finally:
            async with self.lock:
                self.active_tasks -= 1
                await self._process_queue()

    async def get_status(self) -> dict:
        """Get queue status."""
        return {
            "active_tasks": self.active_tasks,
            "queued_tasks": len(self.queue),
            "max_concurrent": self.max_concurrent
        }


# Global ingestion queue: max 2 concurrent ingestions
ingestion_queue = TaskQueue(max_concurrent=2)
