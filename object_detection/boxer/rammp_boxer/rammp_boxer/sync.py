"""Recent frames by timestamp, to pair a detection message with the camera frames it came from."""

from __future__ import annotations

import threading
from collections import OrderedDict


class FrameCache:
    """The last `size` messages keyed by stamp in nanoseconds. Thread-safe."""

    def __init__(self, size: int = 30):
        self.size = size
        self._frames: OrderedDict[int, object] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, stamp_ns: int, msg) -> None:
        with self._lock:
            self._frames[stamp_ns] = msg
            while len(self._frames) > self.size:
                self._frames.popitem(last=False)

    def nearest(self, stamp_ns: int, slop_ns: int):
        """The message closest in time to stamp_ns, if within slop_ns; else None. slop_ns=0 means exact."""
        with self._lock:
            if not self._frames:
                return None
            best = min(self._frames, key=lambda s: abs(s - stamp_ns))
            return self._frames[best] if abs(best - stamp_ns) <= slop_ns else None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
