"""A fabricated operator, for running the pipeline with no camera attached.

Everything downstream of `Observation` -- the retargeting, the filters, the
limits, the sim bridge, the whole UI -- can be exercised from here, which is
what makes the teleop testable on a machine with no webcam, in CI, or while the
camera is busy. `selftest.py` asserts against it; `app.py --source synthetic`
plays the scripted sequence below into the real window.

Landmarks are built in an anatomical body frame

    x = forward (the way you face)    y = your left    z = up

and then rotated into the MediaPipe world convention

    x = image-right    y = image-down    z = away from the camera

by the same torso frame the retargeter will recover, so a pose asked for here
is the pose that comes back out.
"""

from __future__ import annotations

import math

import numpy as np

from .landmarks import H, P, Observation

# Body proportions, metres. Only ratios matter; the retargeter normalises.
SHOULDER_HALF = 0.20
HIP_HALF = 0.12
SHOULDER_Z = 0.50            # above the hip midpoint, which is the origin
UPPER_ARM = 0.30
FOREARM = 0.26
PALM = 0.09

# Orthographic projection into the image, used only for the lift channel: a
# crouch of `crouch` metres moves the shoulders IMAGE_SCALE * crouch down the
# frame. 0.5 puts a 0.44 m squat at the default 0.22 crouch_span.
IMAGE_SCALE = 0.5

DOWN = np.array([0.0, 0.0, -1.0])
LEFT = np.array([0.0, 1.0, 0.0])

_SIDE_SIGN = {"left": 1.0, "right": -1.0}


def _rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector `a` onto unit vector `b`.

    Used to carry the hand around with the arm. A real hand roughly continues
    the forearm and rolls with it, so driving the fake one that way keeps the
    robot's wrist joints near zero through the script -- whereas leaving the
    hand pointing at the floor while the arm swings up demands a wrist rotation
    the robot does not have, and pegs joints 5 to 7 at their limits.
    """
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: turn a half circle about any perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (k @ k)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def arm_dir(fwd_deg: float, out_deg: float, side: str) -> np.ndarray:
    """A limb direction in body coordinates, the way a person would say it.

    fwd_deg  swing forward from hanging straight down (90 = straight ahead)
    out_deg  swing away from the body sideways (90 = straight out to the side)
    """
    s = _SIDE_SIGN[side]
    return _rx(s * math.radians(out_deg)) @ _ry(-math.radians(fwd_deg)) @ DOWN


# The torso frame of someone standing upright, square to the camera, expressed
# in MediaPipe world axes: columns are [forward | left | up].
_T0 = np.column_stack([np.array([0.0, 0.0, -1.0]),    # forward = toward camera
                       np.array([1.0, 0.0, 0.0]),     # their left = image right
                       np.array([0.0, -1.0, 0.0])])   # up = image up


def torso_of(yaw_rad: float) -> np.ndarray:
    """Torso frame after turning `yaw_rad` about your own up axis (+ = to your
    left), in MediaPipe world axes."""
    return _T0 @ _rz(yaw_rad)


def hand_points(approach: np.ndarray, across: np.ndarray,
                pinch_ratio: float) -> np.ndarray:
    """A plausible 21-point hand, in whatever frame `approach`/`across` are in.

    Only the six landmarks the retargeter reads are placed carefully -- wrist,
    the index/middle/pinky knuckles, and the index and thumb tips. The rest are
    filled in so the array is the right shape and the overlay has something to
    draw.
    """
    a = approach / max(np.linalg.norm(approach), 1e-9)
    c = across - float(across @ a) * a
    c = c / max(np.linalg.norm(c), 1e-9)
    # `across` is the jaw axis, which is the palm normal; the knuckles lie
    # along k, chosen so cross(a, pinky - index) gives c back.
    k = np.cross(c, a)
    side = c

    hw = np.zeros((21, 3))
    hw[H.WRIST] = 0.0
    hw[H.INDEX_MCP] = 0.085 * a - 0.040 * k
    hw[H.PINKY_MCP] = 0.085 * a + 0.040 * k
    hw[H.MIDDLE_MCP] = PALM * a
    hw[H.RING_MCP] = 0.088 * a + 0.018 * k
    hw[H.INDEX_TIP] = hw[H.INDEX_MCP] + 0.070 * a
    # The thumb closes across the palm; its distance to the index tip over the
    # palm length is exactly what `retarget.pinch` reads back out.
    hw[H.THUMB_TIP] = hw[H.INDEX_TIP] - (pinch_ratio * PALM) * c
    hw[H.THUMB_CMC] = 0.02 * a - 0.035 * c
    hw[H.THUMB_MCP] = 0.045 * a - 0.050 * c
    hw[H.THUMB_IP] = 0.065 * a - 0.055 * c
    for mcp, pip, dip, tip in ((H.INDEX_MCP, H.INDEX_PIP, H.INDEX_DIP, H.INDEX_TIP),
                               (H.MIDDLE_MCP, H.MIDDLE_PIP, H.MIDDLE_DIP, H.MIDDLE_TIP),
                               (H.RING_MCP, H.RING_PIP, H.RING_DIP, H.RING_TIP),
                               (H.PINKY_MCP, H.PINKY_PIP, H.PINKY_DIP, H.PINKY_TIP)):
        if not hw[tip].any():
            hw[tip] = hw[mcp] + 0.065 * a
        hw[pip] = hw[mcp] + (hw[tip] - hw[mcp]) * 0.40
        hw[dip] = hw[mcp] + (hw[tip] - hw[mcp]) * 0.72
    hw += 0.02 * side          # nudge off the origin; only differences are read
    return hw


def make_observation(t: float, *, yaw: float = 0.0, crouch: float = 0.0,
                     upper: dict | None = None, fore: dict | None = None,
                     approach: dict | None = None, across: dict | None = None,
                     pinch: dict | None = None,
                     with_hands: bool = True) -> Observation:
    """One fabricated tracked frame.

    All directions are in body coordinates (x forward, y your left, z up) and
    need not be normalised. `yaw` is your torso turn in radians, positive to
    your left; `crouch` is how far you have sunk, in metres.
    """
    upper = {"left": DOWN, "right": DOWN} | (upper or {})
    fore = {"left": DOWN, "right": DOWN} | (fore or {})
    approach = {"left": DOWN, "right": DOWN} | (approach or {})
    across = {"left": LEFT, "right": LEFT} | (across or {})
    pinch = {"left": 1.4, "right": 1.4} | (pinch or {})

    tf = torso_of(yaw)

    def to_world(v):
        """Body coordinates -> MediaPipe world axes, for a point or an (n, 3)."""
        v = np.asarray(v, dtype=float)
        return v @ tf.T if v.ndim == 2 else tf @ v

    pw = np.zeros((33, 3))
    body = {
        P.L_SHOULDER: np.array([0.0, SHOULDER_HALF, SHOULDER_Z]),
        P.R_SHOULDER: np.array([0.0, -SHOULDER_HALF, SHOULDER_Z]),
        P.L_HIP: np.array([0.0, HIP_HALF, 0.0]),
        P.R_HIP: np.array([0.0, -HIP_HALF, 0.0]),
        P.NOSE: np.array([0.05, 0.0, SHOULDER_Z + 0.25]),
    }
    hand_world, hand_image = {}, {}
    for side, (sh_i, el_i, wr_i, ix_i, pk_i, th_i) in (
            ("left", (P.L_SHOULDER, P.L_ELBOW, P.L_WRIST, P.L_INDEX, P.L_PINKY, P.L_THUMB)),
            ("right", (P.R_SHOULDER, P.R_ELBOW, P.R_WRIST, P.R_INDEX, P.R_PINKY, P.R_THUMB))):
        u = np.asarray(upper[side], float)
        u = u / max(np.linalg.norm(u), 1e-9)
        f = np.asarray(fore[side], float)
        f = f / max(np.linalg.norm(f), 1e-9)
        a = np.asarray(approach[side], float)
        a = a / max(np.linalg.norm(a), 1e-9)
        c = np.asarray(across[side], float)

        sh = body[sh_i]
        el = sh + UPPER_ARM * u
        wr = el + FOREARM * f
        body[el_i] = el
        body[wr_i] = wr
        # The pose model's own coarse hand points, so the pose-only fallback in
        # retarget.hand_dirs has something consistent to read.
        c_perp = c - float(c @ a) * a
        c_perp = c_perp / max(np.linalg.norm(c_perp), 1e-9)
        k = np.cross(c_perp, a)                 # knuckle line; c_perp is the jaw
        body[ix_i] = wr + 0.085 * a - 0.040 * k
        body[pk_i] = wr + 0.085 * a + 0.040 * k
        body[th_i] = wr + 0.050 * a - 0.055 * c_perp

        if with_hands:
            hand_world[side] = to_world(hand_points(a, c_perp, pinch[side]))
        else:
            hand_world[side] = None

    # Only the landmarks this fake body actually has are marked visible. The
    # rest stay at the origin, and zero visibility is what stops the overlay
    # drawing a star of bones out to the middle of the frame.
    vis = np.zeros(33)
    for i, p in body.items():
        pw[i] = to_world(p)
        vis[i] = 1.0

    # Orthographic image projection. Crouching lowers the body in the room,
    # which the hip-centred world landmarks cannot show, so it is added here --
    # this is exactly the asymmetry the lift channel exploits.
    pi = np.zeros((33, 3))
    pi[:, 0] = 0.5 + pw[:, 0] * IMAGE_SCALE
    pi[:, 1] = 0.5 + (pw[:, 1] + crouch) * IMAGE_SCALE
    pi[:, 2] = pw[:, 2]

    for side, hw in hand_world.items():
        if hw is None:
            hand_image[side] = None
            continue
        wr_i = P.L_WRIST if side == "left" else P.R_WRIST
        # Hand world landmarks are hand-centred; re-anchor them on the tracked
        # wrist so the drawn hand lands on the drawn arm.
        rel = hw - hw[H.WRIST]
        him = np.zeros((21, 3))
        him[:, 0] = pi[wr_i, 0] + rel[:, 0] * IMAGE_SCALE
        him[:, 1] = pi[wr_i, 1] + rel[:, 1] * IMAGE_SCALE
        hand_image[side] = him

    return Observation(t=t, pose_world=pw, pose_image=pi,
                       pose_vis=vis, hand_world=hand_world,
                       hand_image=hand_image)


# --------------------------------------------------------------------------
# a scripted operator
# --------------------------------------------------------------------------
def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# (time_s, label, left(fwd,out), right(fwd,out), yaw_deg, crouch_m, pinch)
SCRIPT = [
    (0.0, "rest", (0, 0), (0, 0), 0, 0.00, 1.4),
    (3.0, "left arm forward", (90, 0), (0, 0), 0, 0.00, 1.4),
    (6.0, "right arm out", (90, 0), (0, 85), 0, 0.00, 1.4),
    (9.0, "both forward, elbows bent", (75, 10), (75, 10), 0, 0.00, 1.4),
    (12.0, "crouch", (75, 10), (75, 10), 0, 0.44, 1.4),
    (15.0, "turn left", (75, 10), (75, 10), 35, 0.20, 1.4),
    (18.0, "turn right", (75, 10), (75, 10), -35, 0.20, 1.4),
    (21.0, "close grippers", (80, 5), (80, 5), 0, 0.00, 0.30),
    (24.0, "rest", (0, 0), (0, 0), 0, 0.00, 1.4),
]
SCRIPT_LENGTH = SCRIPT[-1][0]


def scripted(t: float, loop: bool = True) -> tuple[Observation, str]:
    """The operator at time `t` through the sequence above, plus its label."""
    if loop:
        t = math.fmod(t, SCRIPT_LENGTH)
    i = 0
    while i + 1 < len(SCRIPT) - 1 and t >= SCRIPT[i + 1][0]:
        i += 1
    t0, label, l0, r0, y0, c0, p0 = SCRIPT[i]
    t1, _, l1, r1, y1, c1, p1 = SCRIPT[i + 1]
    a = _smoothstep((t - t0) / max(t1 - t0, 1e-6))

    def lerp(x, y):
        return x + (y - x) * a

    lf, lo = lerp(l0[0], l1[0]), lerp(l0[1], l1[1])
    rf, ro = lerp(r0[0], r1[0]), lerp(r0[1], r1[1])
    yaw, crouch, pin = lerp(y0, y1), lerp(c0, c1), lerp(p0, p1)

    # Bend the elbow in proportion to how far the arm is raised, so the script
    # exercises joints 3 and 4 rather than sweeping a rigid stick around.
    def limb(fwd, out, side):
        u = arm_dir(fwd, out, side)
        f = arm_dir(fwd + 0.45 * fwd, out, side)
        return u, f

    ul, fl = limb(lf, lo, "left")
    ur, fr = limb(rf, ro, "right")
    # The hand continues the forearm and is carried round with it, so the wrist
    # joints stay near zero instead of being asked for rotations the robot has
    # no joint for.
    hands = {s: align(DOWN, f) for s, f in (("left", fl), ("right", fr))}
    obs = make_observation(
        t, yaw=math.radians(yaw), crouch=crouch,
        upper={"left": ul, "right": ur}, fore={"left": fl, "right": fr},
        approach={"left": fl, "right": fr},
        across={s: R @ LEFT for s, R in hands.items()},
        pinch={"left": pin, "right": pin})
    return obs, label
