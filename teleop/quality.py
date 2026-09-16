"""Is this pose fit to calibrate from?

Calibration freezes whatever you are doing as the robot's zero. That makes a bad
calibration pose invisible afterwards: everything still tracks perfectly, just
against a crooked reference, and the robot sits at a yaw offset you cannot
explain.

The way it actually happens is reaching for the keyboard. Leaning toward the
machine to press C turns one shoulder forward, and a rotated shoulder line IS
torso yaw as far as the mapping is concerned -- so the zero gets taken with you
mid-twist. A mouse click has exactly the same problem, because you lean the same
way to reach the mouse.

So the fix is not a different button. It is calibrating while nobody is touching
anything, and refusing to do it from a pose that is obviously wrong. This module
is the refusing part.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .landmarks import P
from .retarget import torso_frame, unit

# Stillness is judged on the TORSO only, not the arms.
#
# Watching the wrists too was a mistake, and the recorded session shows why:
# only 6% of ticks passed, because landmark jitter on a fast-moving hand is
# several times the torso's, and the tracker flickers even when you do not move
# at all. Calibration reads the torso frame; whether a hand twitched while it
# happened does not matter.
_WATCH = (P.L_SHOULDER, P.R_SHOULDER, P.L_HIP, P.R_HIP)

# MediaPipe world axes put -z toward the camera, so this is "facing the lens".
_TOWARD_CAMERA = np.array([0.0, 0.0, -1.0])


class PoseCheck:
    """Rolling judgement on whether the operator is ready to be calibrated."""

    # Thresholds measured off a real 79-second session rather than guessed.
    # The first set were guesses and they blocked calibration completely:
    # `still` passed 6% of ticks and `square` 72%, so the countdown never once
    # reached zero and the robot was driven for 0% of the session.
    #
    # `square` is the weakest signal of the four. It comes out of MediaPipe's
    # world-landmark z, which is its least reliable axis, and it read a median
    # of 15 deg with excursions past 50 deg from someone largely facing the
    # camera. 30 deg passes 90% of that session while still catching a genuine
    # quarter turn, which is all it is really there for.
    def __init__(self, window: float = 0.7, move_thresh: float = 0.05,
                 level_thresh: float = 0.06, square_deg: float = 30.0,
                 arms_down_margin: float = 0.10):
        self.window = window
        self.move_thresh = move_thresh
        self.level_thresh = level_thresh
        self.square_deg = square_deg
        self.arms_down_margin = arms_down_margin
        self._hist: deque = deque()

    def reset(self) -> None:
        self._hist.clear()

    def update(self, obs, min_vis: float = 0.5) -> dict:
        """Check the current frame. Returns a dict of named booleans plus `ok`.

        Each check is reported separately rather than as one verdict, because
        "not ready" on its own tells the operator nothing they can act on.
        """
        out = dict(seen=False, still=False, level=False, square=False,
                   arms_down=False, ok=False, motion=0.0, tilt_deg=0.0,
                   turn_deg=0.0)
        if obs is None or not obs.visible(min_vis):
            self._hist.clear()
            return out
        out["seen"] = True

        pi, pw = obs.pose_image, obs.pose_world

        # ---- stillness, in image space over a short window ----------------
        pts = np.array([pi[i][:2] for i in _WATCH], dtype=float)
        self._hist.append((obs.t, pts))
        while self._hist and obs.t - self._hist[0][0] > self.window:
            self._hist.popleft()
        if len(self._hist) >= 3:
            stack = np.stack([p for _, p in self._hist])
            # Worst wander of any watched landmark across the window.
            motion = float(np.max(stack.max(axis=0) - stack.min(axis=0)))
            out["motion"] = motion
            out["still"] = motion <= self.move_thresh
        else:
            out["motion"] = float("inf")

        # ---- shoulders level ----------------------------------------------
        tilt = float(pi[P.L_SHOULDER][1] - pi[P.R_SHOULDER][1])
        out["tilt_deg"] = math.degrees(math.atan2(
            tilt, max(abs(pi[P.L_SHOULDER][0] - pi[P.R_SHOULDER][0]), 1e-6)))
        out["level"] = abs(tilt) <= self.level_thresh

        # ---- square to the camera -----------------------------------------
        # This is the one that leaning breaks, and the one that turns into a
        # phantom yaw offset for the rest of the session.
        frame = torso_frame(pw, obs.hips_visible(min_vis))
        if frame is not None:
            fwd = unit(frame[:, 0])
            turn = math.degrees(math.acos(
                float(np.clip(fwd @ _TOWARD_CAMERA, -1.0, 1.0))))
            out["turn_deg"] = turn
            out["square"] = turn <= self.square_deg

        # ---- arms hanging ---------------------------------------------------
        sh_y = 0.5 * (pi[P.L_SHOULDER][1] + pi[P.R_SHOULDER][1])
        wr_y = min(pi[P.L_WRIST][1], pi[P.R_WRIST][1])
        out["arms_down"] = bool(wr_y > sh_y + self.arms_down_margin)

        out["ok"] = bool(out["still"] and out["level"] and out["square"]
                         and out["arms_down"])
        return out


def issues(checks: dict) -> list[str]:
    """What to tell the operator to change, most important first."""
    if not checks.get("seen"):
        return ["step back so your head and shoulders are in frame"]
    msgs = []
    if not checks.get("square"):
        msgs.append(f"turn square to the camera ({checks['turn_deg']:.0f} deg off)")
    if not checks.get("level"):
        msgs.append("level your shoulders")
    if not checks.get("arms_down"):
        msgs.append("let your arms hang down")
    if not checks.get("still"):
        msgs.append("hold still")
    return msgs
