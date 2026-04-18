"""Batch prefetch utilities for overlapping host work with JAX execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import jax


@dataclass(frozen=True)
class DeviceBatchPrefetcher:
  """Moves a small queue of batches onto the target device ahead of use."""

  prefetch_batches: int = 2

  def put(self, batch: Any) -> Any:
    return jax.tree_util.tree_map(jax.device_put, batch)

  def iter_batches(self, batches: Iterable[Any]) -> Iterator[Any]:
    if self.prefetch_batches <= 0:
      yield from batches
      return

    queue = deque()
    for batch in batches:
      queue.append(self.put(batch))
      if len(queue) > self.prefetch_batches:
        yield queue.popleft()

    while queue:
      yield queue.popleft()
