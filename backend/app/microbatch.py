# backend/app/microbatch.py
import asyncio
import time
from typing import Any, Callable, List, Tuple

class MicroBatcher:
    def __init__(self, infer_fn: Callable[[List[str]], Any],
                 max_batch: int = 128, max_wait_ms: int = 8):
        self.infer_fn = infer_fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self.queue: asyncio.Queue[Tuple[str, asyncio.Future]] = asyncio.Queue()
        self._task = asyncio.create_task(self._loop())

    async def submit(self, item: str):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _loop(self):
        while True:
            item, fut = await self.queue.get()  # ждём первый элемент партии
            batch = [item]; futures = [fut]
            start = time.perf_counter()

            # добираем до max_batch или пока не истекло окно
            while len(batch) < self.max_batch:
                timeout = self.max_wait - (time.perf_counter() - start)
                if timeout <= 0:
                    break
                try:
                    item2, fut2 = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(item2); futures.append(fut2)
                except asyncio.TimeoutError:
                    break

            try:
                results = await self.infer_fn(batch)
            except Exception as e:
                for f in futures:
                    if not f.done(): f.set_exception(e)
                continue

            for f, res in zip(futures, results):
                if not f.done(): f.set_result(res)

    async def close(self):
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
