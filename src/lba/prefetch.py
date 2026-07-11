"""Background prefetching for planned batches."""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ProducerError:
    error: BaseException


def prefetch_iterator(source: Generator[Any, None, None], max_batches: int) -> Iterator[Any]:
    """Yield from source through a bounded producer queue."""

    if max_batches <= 0:
        yield from source
        return

    prefetch_queue: queue.Queue[Any] = queue.Queue(maxsize=max_batches)
    stop_event = threading.Event()
    done_sentinel = object()

    def enqueue(item: Any) -> bool:
        while not stop_event.is_set():
            try:
                prefetch_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce_batches() -> None:
        try:
            for batch in source:
                if stop_event.is_set():
                    break
                if not enqueue(batch):
                    break
        except BaseException as error:
            enqueue(_ProducerError(error))
        finally:
            try:
                source.close()
            except RuntimeError:
                pass
            enqueue(done_sentinel)

    producer_thread = threading.Thread(
        target=produce_batches,
        name="lba-prefetch",
        daemon=True,
    )
    producer_thread.start()

    try:
        while True:
            queued_item = prefetch_queue.get()
            if queued_item is done_sentinel:
                break
            if isinstance(queued_item, _ProducerError):
                raise queued_item.error
            yield queued_item
    finally:
        stop_event.set()
        producer_thread.join(timeout=1)
