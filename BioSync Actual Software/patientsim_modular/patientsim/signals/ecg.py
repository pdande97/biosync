import numpy as np
from .base import SignalSource

class ECG(SignalSource):
    """ECG: P half-sine, QRS with Q-/R+/S-, flat delays, T full half-sine.
    Inputs: fs (Hz), rate (BPM), amp_p (mV), amp_r (mV), amp_t (mV),
            p_ms, qrs_ms, t_ms, t1_ms, t2_ms.
    """
    def __init__(self):
        self.fs = 250
        self.bpm = 60.0
        self.amp_p = 2.0     # P peak (mV)
        self.amp_r = 5.0     # R peak (mV)
        self.amp_t = 3.0     # T peak (mV)
        self.p_ms = 80.0
        self.qrs_ms = 100.0
        self.t_ms = 160.0
        self.t1_ms = 80.0
        self.t2_ms = 120.0

        self._beat = np.zeros(1, dtype=float)
        self._i = 0
        self._period = 1

    def _half_sine(self, N, amp):
        """Return a full half-sine from 0→amp→0 with N samples, including both endpoints."""
        N = max(2, int(N))
        idx = np.arange(N)
        # goes 0 at idx=0, amp at middle, 0 at idx=N-1
        return amp * np.sin(np.pi * idx / (N - 1))

    def configure(self, *, fs: int, rate: float, amplitude: float = 1.0, **kw) -> None:
        self.fs = int(fs)
        self.bpm = float(rate)
        self.amp_p = float(kw.get("amp_p", self.amp_p))
        self.amp_r = float(kw.get("amp_r", kw.get("amp_qrs", self.amp_r)))
        self.amp_t = float(kw.get("amp_t", self.amp_t))
        self.p_ms   = float(kw.get("p_ms", self.p_ms))
        self.qrs_ms = float(kw.get("qrs_ms", self.qrs_ms))
        self.t_ms   = float(kw.get("t_ms", self.t_ms))
        self.t1_ms  = float(kw.get("t1_ms", self.t1_ms))
        self.t2_ms  = float(kw.get("t2_ms", self.t2_ms))

        # sample counts
        p_len   = max(2, int(round(self.fs * self.p_ms   / 1000.0)))
        t1_len  = max(0, int(round(self.fs * self.t1_ms  / 1000.0)))
        qrs_len = max(3, int(round(self.fs * self.qrs_ms / 1000.0)))  # >=3 for Q,R,S
        t2_len  = max(0, int(round(self.fs * self.t2_ms  / 1000.0)))
        t_len   = max(2, int(round(self.fs * self.t_ms   / 1000.0)))

        period_samples = max(1, int(round((60.0 / self.bpm) * self.fs)))

        total = p_len + t1_len + qrs_len + t2_len + t_len
        if total > period_samples:
            scale = period_samples / float(total)
            p_len   = max(2, int(round(p_len   * scale)))
            t1_len  = max(0, int(round(t1_len  * scale)))
            qrs_len = max(3, int(round(qrs_len * scale)))
            t2_len  = max(0, int(round(t2_len  * scale)))
            t_len   = max(2, int(round(t_len   * scale)))
            total = p_len + t1_len + qrs_len + t2_len + t_len
        rest_len = max(0, period_samples - total)

        # --- Build morphology ---
        # P: full half-sine
        p = self._half_sine(p_len, self.amp_p)

        # QRS: Q (neg), R (pos), S (neg) using proportions; auto depths from R
        q_len = max(1, int(round(qrs_len * 0.18)))
        r_len = max(1, int(round(qrs_len * 0.34)))
        s_len = max(1, qrs_len - q_len - r_len)

        q_depth = 0.20 * self.amp_r   # ~20% of R
        s_depth = 0.45 * self.amp_r   # ~45% of R

        q = np.linspace(0.0, -q_depth, q_len, endpoint=False)
        r = np.linspace(-q_depth, self.amp_r, r_len, endpoint=False)
        s_down = np.linspace(self.amp_r, -s_depth, max(1, s_len//2), endpoint=False)
        s_up   = np.linspace(-s_depth, 0.0, s_len - len(s_down), endpoint=False)
        qrs = np.concatenate([q, r, s_down, s_up])

        # Flat isoelectric delays
        z1 = np.zeros(t1_len, dtype=float)
        z2 = np.zeros(t2_len, dtype=float)

        # T: full half-sine (0→amp→0)
        t = self._half_sine(t_len, self.amp_t)

        rest = np.zeros(rest_len, dtype=float)

        self._beat = np.concatenate([p, z1, qrs, z2, t, rest]).astype(float)
        self._period = len(self._beat)
        self._i = 0

    def step(self, n: int):
        out = []
        L = self._period
        for _ in range(n):
            y = float(self._beat[self._i % L])
            out.append(y)
            self._i += 1
        return out
