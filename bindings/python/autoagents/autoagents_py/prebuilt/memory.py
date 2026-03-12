"""Prebuilt memory providers."""

from __future__ import annotations

from dataclasses import dataclass

from .._core import sliding_window_memory


@dataclass(slots=True)
class SlidingWindowMemory:
    window_size: int = 20

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")

    def _to_core_provider(self):
        return sliding_window_memory(self.window_size)
