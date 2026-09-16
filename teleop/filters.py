"""Signal conditioning for the tracking stream.

Landmark estimates are noisy at rest and laggy when you move fast, and a plain
exponential average cannot fix both: the smoothing constant that kills the
jitter also adds the lag. The One Euro filter adapts instead -- it smooths hard
when the signal is slow and barely at all when it is fast -- which is what makes
hand tracking feel attached to the hand rather than trailing it.

Reference: Casiez, Roussel, Vogel, "1 Euro Filter: A Simple Speed-based Low-pass
Filter for Noisy Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at `cutoff` Hz, sampled at dt."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One scalar (or one fixed-size vector) of One Euro filtering.

    mincutoff  cutoff in Hz when the signal is still. Lower = steadier hands,
               more lag. This is the knob to turn first.
    beta       how much the cutoff opens up with speed. Higher = less lag when
               you move fast, more jitter passed through.
    dcutoff    cutoff of the derivative estimate itself. Rarely worth changing.
    """

    def __init__(self, mincutoff: float = 1.2, beta: float = 0.05,
                 dcutoff: float = 1.0, shape: tuple[int, ...] | None = None):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x_prev = None if shape is None else np.zeros(shape)
        self._dx_prev = None if shape is None else np.zeros(shape)
        self._t_prev: float | None = None
        self._primed = False

    def reset(self) -> None:
        self._primed = False
        self._t_prev = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if not self._primed:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            self._primed = True
            return x.copy()

        dt = t - self._t_prev
        # A stalled or rewound clock would make the filter explode; fall back to
        # a nominal 60 Hz rather than dividing by ~0.
        if not (1e-4 < dt < 1.0):
            dt = 1.0 / 60.0
        self._t_prev = t

        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        # _alpha is scalar maths but numpy broadcasts it elementwise, so a
        # vector signal gets a per-component adaptive cutoff for free.
        a = 1.0 / (1.0 + (1.0 / (2.0 * math.pi * cutoff)) / dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat.copy()


class AngleOneEuro(OneEuro):
    """One Euro on an angle, filtering the unwrapped signal.

    Filtering raw radians puts a spike through the output every time the signal
    crosses +/-pi. This tracks the unwrapped value and wraps only on the way
    out, so a yaw that walks past pi stays continuous.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._offset = 0.0
        self._last_raw = 0.0

    def __call__(self, x, t: float):
        x = float(x)
        if self._primed:
            # accumulate whole turns so the filtered signal never jumps
            d = x - self._last_raw
            if d > math.pi:
                self._offset -= 2.0 * math.pi
            elif d < -math.pi:
                self._offset += 2.0 * math.pi
        self._last_raw = x
        y = float(super().__call__(x + self._offset, t))
        return math.atan2(math.sin(y), math.cos(y))


class RateLimit:
    """Cap how fast a command may change, in units per second.

    The filters above smooth noise; this bounds the worst case. A tracking
    dropout that snaps a joint from one limit to the other is a plausible
    failure, and on a position actuator it becomes a full-torque lunge. This
    turns it into a slew.
    """

    def __init__(self, max_rate: float, shape: tuple[int, ...] | None = None):
        self.max_rate = max_rate
        self._y = None if shape is None else np.zeros(shape)
        self._t: float | None = None

    def reset(self, value=None) -> None:
        """Forget the clock, but keep `value` as where the output currently is.

        The distinction matters. A limiter rebuilt with no memory passes its
        first command straight through, because it has nothing to limit against
        -- and the first command after a recalibration can be a full-scale step.
        Measured in a real session: recalibrating at the bottom of a crouch
        stepped the lift command 0.000 -> 0.700 m in a single tick, 9.998 m/s
        against a 0.6 m/s limit, and the carriage lunged the whole travel. Seed
        it with where the robot actually is and the step becomes a slew again.
        """
        self._y = None if value is None else np.asarray(value, dtype=float).copy()
        self._t = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._y is None:
            # Genuinely nothing to limit against: adopt the first command.
            self._y, self._t = x.copy(), t
            return x.copy()
        if self._t is None:
            # Seeded but not yet clocked: hold this tick, start limiting next.
            self._t = t
            return self._y.copy()
        dt = max(1e-4, min(t - self._t, 0.5))
        self._t = t
        step = self.max_rate * dt
        self._y = self._y + np.clip(x - self._y, -step, step)
        return self._y.copy()
