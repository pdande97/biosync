import numpy as np
from .base import SignalSource

class Respiration(SignalSource):
    """Asymmetric inhale/exhale waveform (breaths per minute)."""
    def __init__(self):
        self.fs = 100
        self.brpm = 12.0
        self.amp = 1.0
        self._cycle = None
        self._i = 0
        self._period = 1

    def configure(self, *, fs: int, rate: float, amplitude: float = 1.0) -> None:
        self.fs, self.brpm, self.amp = fs, rate, amplitude
        N = fs  # 1-second base shape, we will stretch by period
        t = np.linspace(0, 1, N, endpoint=False)
        # quick asymmetry: faster rise (inhale), slower decay (exhale)
        inhale = 1 - np.exp(-5*t)
        exhale = np.exp(-3*t)
        shape = inhale * exhale
        # normalize
        shape = shape / np.max(np.abs(shape))
        self._cycle = shape
        self._period = max(1, int((60.0 / self.brpm) * self.fs))
        self._i = 0

    def step(self, n: int):
        out = []
        L = len(self._cycle)
        for _ in range(n):
            if self._i >= self._period:
                self._i = 0
            out.append(self.amp * float(self._cycle[self._i % L]))
            self._i += 1
        return out
