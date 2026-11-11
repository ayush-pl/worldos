"""Named, reproducible random streams.

Each engine draws from its own stream derived from (seed, stream name), so
adding or reordering an engine never perturbs the random numbers seen by
another engine. Experiments use common random numbers: every arm of an
experiment is reseeded identically at the branch point.
"""

from __future__ import annotations

import zlib

import numpy as np


class RandomStreams:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._streams: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        gen = self._streams.get(name)
        if gen is None:
            gen = np.random.default_rng([self.seed, zlib.crc32(name.encode("utf-8"))])
            self._streams[name] = gen
        return gen

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._streams.clear()
