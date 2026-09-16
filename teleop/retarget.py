"""Turn a tracked human into robot joint commands.

This module is pure numpy on purpose: no mediapipe, no mujoco, no cv2. It takes
an `Observation` full of landmark arrays and returns `Targets` full of numbers.
That is what lets `selftest.py` check the maths against MuJoCo's own forward
kinematics, and `synthetic.py` drive the whole app with no camera attached.

--------------------------------------------------------------------------
Why angle retargeting and not inverse kinematics
--------------------------------------------------------------------------
The obvious approach is to track your wrist and run IK on the end effector.
It is the wrong one here. The OpenArm is 7-DOF with a tightly limited wrist
(joint6 is +/-45 deg, joint4 cannot go below 0), so a position-only IK solve has
a null space it will wander through, and it gives no way to say "my elbow is
out". It also cannot be verified offline.

The arm is built anatomically -- shoulder pitch, shoulder abduction, humeral
twist, elbow, forearm twist, wrist pitch, wrist roll -- so your arm's joint
angles map onto it directly. Those angles come out in closed form from three
landmarks per arm: exact, no iteration to diverge, no null space to drift in,
and it round-trips against MuJoCo's own FK to ~1e-12 (see selftest.py).

--------------------------------------------------------------------------
The two frames, which are deliberately the same
--------------------------------------------------------------------------
The arm's base frame was measured off the compiled model (selftest re-checks it
every run):

    x = forward, out of the robot's chest     y = the robot's left     z = up

The torso frame built from your shoulders and hips below uses exactly that
convention, so a direction expressed in your torso frame IS a direction in the
arm's base frame. No conversion, nothing to get backwards.

--------------------------------------------------------------------------
The joint chain, read off the model
--------------------------------------------------------------------------
Every arm link has an identity body quaternion -- the links are pure
translations -- so the joints compose as plain rotations about these axes, each
given in its own link's frame:

    joint1  left +y, right -y   shoulder pitch      (swing the arm forward/back)
    joint2  -x both sides       shoulder abduction  (swing it out to the side)
    joint3  -z both sides       humeral twist       (sets the elbow's plane)
    joint4  -y both sides       elbow               (0..140 deg, flexion only)
    joint5  -z both sides       forearm twist
    joint6  left -y, right +y   wrist pitch         (+/-45 deg only)
    joint7  +x both sides       wrist roll

The upper arm and the forearm both run along their link's local -z, which is
what makes the closed form below short.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import TeleopConfig
from .filters import AngleOneEuro, OneEuro, RateLimit
from .landmarks import H, P, POSE_ARM, SIDES, Observation

# Joint limits as compiled into the model. `Retargeter` prefers the ones read
# off the live MjModel; these are the fallback, so the maths stays testable with
# no MuJoCo present.
DEFAULT_LIMITS = {
    "left": np.array([[-3.4907, 1.3963], [-3.3161, 0.1745], [-1.5708, 1.5708],
                      [0.0, 2.4435], [-1.5708, 1.5708], [-0.7854, 0.7854],
                      [-1.5708, 1.5708]]),
    "right": np.array([[-1.3963, 3.4907], [-0.1745, 3.3161], [-1.5708, 1.5708],
                       [0.0, 2.4435], [-1.5708, 1.5708], [-0.7854, 0.7854],
                       [-1.5708, 1.5708]]),
}

_DOWN = np.array([0.0, 0.0, -1.0])      # every link points along its own -z
_JAW = np.array([0.0, 1.0, 0.0])        # the jaws separate along gripper +y


# --------------------------------------------------------------------------
# small rotation helpers
# --------------------------------------------------------------------------
def rx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def ry(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def unit(v, fallback=_DOWN) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return np.asarray(fallback, dtype=float) if n < 1e-9 else v / n


def _sgn1(side: str) -> float:
    """joint1's axis is +y on the left arm and -y on the right."""
    return 1.0 if side == "left" else -1.0


def _sgn6(side: str) -> float:
    """joint6's axis is -y on the left arm and +y on the right."""
    return -1.0 if side == "left" else 1.0


# --------------------------------------------------------------------------
# forward kinematics of one arm, in the arm's base frame
# --------------------------------------------------------------------------
def arm_fk(q, side: str) -> dict:
    """Rotations and link directions for one arm, from joint angles alone.

    All in the arm's base frame. selftest.py checks every field of this against
    MuJoCo, which is the only reason to trust the inverse below.
    """
    q = np.asarray(q, dtype=float)
    s1, s6 = _sgn1(side), _sgn6(side)
    a1, a2 = ry(s1 * q[0]), rx(-q[1])
    a3, a4 = rz(-q[2]), ry(-q[3])
    a5, a6, a7 = rz(-q[4]), ry(s6 * q[5]), rx(q[6])

    r12 = a1 @ a2                       # link2's frame, and the upper arm's
    r04 = r12 @ a3 @ a4                 # link4's frame, and the forearm's
    r07 = r04 @ a5 @ a6 @ a7            # the gripper's frame
    return dict(
        upper=r12 @ _DOWN,              # shoulder -> elbow, unit
        fore=r04 @ _DOWN,               # elbow -> wrist, unit
        approach=r07 @ _DOWN,           # out of the jaws (gripper local -z)
        jaw=r07 @ _JAW,                 # across the jaws (gripper local +y)
        r12=r12, r04=r04, r07=r07,
    )


# --------------------------------------------------------------------------
# inverse: directions -> joint angles, in closed form
# --------------------------------------------------------------------------
def solve_shoulder_elbow(upper, fore, side: str,
                         prev: np.ndarray | None = None) -> np.ndarray:
    """joints 1-4 from the upper-arm and forearm directions.

    For the left arm (the right differs only in joint1's sign):

        upper = Ry(q1) Rx(-q2) (0,0,-1)
              = (-cos q2 sin q1,  -sin q2,  -cos q2 cos q1)

    so sin q2 = -upper_y, and (sin q1, cos q1) is proportional to
    (-upper_x, -upper_z) with a common factor cos q2 >= 0 that atan2 divides
    out. joint3 is absent because it turns about the upper arm's own axis and
    so cannot move it.

    Rotating the forearm back into link2's frame leaves

        b = Rz(-q3) Ry(-q4) (0,0,-1)
          = (sin q4 cos q3,  -sin q4 sin q3,  -cos q4)

    which reads off as q4 = acos(-b_z) and q3 = atan2(-b_y, b_x).

    Branch: asin puts q2 in +/-90 deg, and (q1, q2) with q2 there already covers
    every direction on the sphere, so no arm direction is unreachable. The arm
    can physically take q2 out to -190 deg, and those poses have a second set of
    angles that points the arm the same way -- this always returns the +/-90 deg
    one. That is the right choice rather than a limitation: a human shoulder
    does not abduct past 90 deg either, so the other branch is never what the
    operator meant, and never picking it keeps the elbow from flipping over
    mid-motion when the tracker jitters across the boundary.
    """
    u, f = unit(upper), unit(fore)
    s1 = _sgn1(side)

    q2 = -math.asin(float(np.clip(u[1], -1.0, 1.0)))
    if math.cos(q2) < 1e-4:
        # Arm straight out to the side: joint1 now turns about the arm's own
        # axis and cannot move it. Hold the last value rather than let atan2 of
        # two near-zeros pick an arbitrary one and snap the shoulder round.
        q1 = float(prev[0]) if prev is not None else 0.0
    else:
        q1 = math.atan2(-s1 * u[0], -u[2])

    b = (ry(s1 * q1) @ rx(-q2)).T @ f
    q4 = math.acos(float(np.clip(-b[2], -1.0, 1.0)))
    if math.sin(q4) < 1e-4:
        # Elbow straight: the twist that would set its plane is unobservable.
        q3 = float(prev[2]) if prev is not None else 0.0
    else:
        q3 = math.atan2(-b[1], b[0])
    return np.array([q1, q2, q3, q4], dtype=float)


def solve_wrist(d: np.ndarray, side: str,
                prev: np.ndarray | None = None) -> np.ndarray:
    """joints 5-7 from the gripper rotation `d`, expressed in link4's frame.

        d = Rz(-q5) Ry(s6 q6) Rx(q7)

    a standard Z-Y-X Euler triple once the axis signs are pulled out front.
    """
    s6 = _sgn6(side)
    beta = math.asin(float(np.clip(-d[2, 0], -1.0, 1.0)))
    if abs(math.cos(beta)) < 1e-6:
        # Gimbal lock: q5 and q7 now turn about the same axis, so the split
        # between them is arbitrary. Put it all in q5 and leave q7 where it was.
        alpha = math.atan2(-d[0, 1], d[1, 1])
        gamma = float(prev[2]) if prev is not None else 0.0
    else:
        alpha = math.atan2(d[1, 0], d[0, 0])
        gamma = math.atan2(d[2, 1], d[2, 2])
    return np.array([-alpha, s6 * beta, gamma], dtype=float)


def solve_arm(upper, fore, r_gripper: np.ndarray | None, side: str,
              prev: np.ndarray | None = None) -> np.ndarray:
    """All seven joints of one arm.

    `r_gripper` is the desired gripper frame in the arm's base frame; pass None
    to leave the three wrist joints at zero.
    """
    q14 = solve_shoulder_elbow(upper, fore, side, prev)
    if r_gripper is None:
        return np.concatenate([q14, np.zeros(3)])
    s1 = _sgn1(side)
    r04 = ry(s1 * q14[0]) @ rx(-q14[1]) @ rz(-q14[2]) @ ry(-q14[3])
    q57 = solve_wrist(r04.T @ r_gripper, side,
                      None if prev is None else prev[4:7])
    return np.concatenate([q14, q57])


def gripper_frame(approach, across) -> np.ndarray:
    """A gripper rotation from the two directions that define it.

    The jaws reach out along the frame's -z and separate along its +y; both were
    measured off the compiled model. `across` only has to be roughly
    perpendicular to `approach` -- it is orthogonalised here.
    """
    z = -unit(approach)
    y = np.asarray(across, dtype=float)
    y = y - float(y @ z) * z
    if np.linalg.norm(y) < 1e-6:                # across was parallel to approach
        alt = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = alt - float(alt @ z) * z
    y = unit(y, [0.0, 1.0, 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def _wrap_into(q: float, lo: float, hi: float) -> float:
    """Clamp, but try the +/-360 deg aliases first.

    joint1 runs to -200 deg on the left arm, so a raw atan2 result of +170 deg
    names the same pose as -190 deg, which is reachable. Clamping it straight to
    +80 would throw away an overhead reach for nothing.
    """
    for cand in (q, q - 2.0 * math.pi, q + 2.0 * math.pi):
        if lo <= cand <= hi:
            return cand
    return float(np.clip(q, lo, hi))


def clamp_to_limits(q, limits: np.ndarray) -> np.ndarray:
    return np.array([_wrap_into(float(v), float(lo), float(hi))
                     for v, (lo, hi) in zip(q, limits)], dtype=float)


# --------------------------------------------------------------------------
# the human side
# --------------------------------------------------------------------------
# MediaPipe's world axes are camera-aligned with +y pointing DOWN the image, so
# this is "up" for an operator who is upright in front of the camera.
CAMERA_UP = np.array([0.0, -1.0, 0.0])


def torso_frame(pose_world: np.ndarray, hips_ok: bool = True) -> np.ndarray | None:
    """Columns [forward | left | up] of your torso, in MediaPipe world axes.

    `up` is taken as the primary axis because it is the best conditioned of the
    three: the shoulder-to-hip span is long and lies in the image plane, where
    tracking is strongest, whereas `forward` comes out of a cross product and
    inherits whatever the noisy depth axis is doing.

    With `hips_ok` false -- you are sitting at a desk, and the hips are cropped
    or behind it -- the camera's own up axis stands in. MediaPipe still emits
    hip landmarks in that case, but they are extrapolated off the bottom of the
    frame and tilt the whole frame with them, which is worse than assuming you
    are upright. The cost is that leaning is no longer tracked; the benefit is
    that everything else works from a chair.
    """
    sh = 0.5 * (pose_world[P.L_SHOULDER] + pose_world[P.R_SHOULDER])
    lat = np.asarray(pose_world[P.L_SHOULDER] - pose_world[P.R_SHOULDER], dtype=float)
    if hips_ok:
        hip = 0.5 * (pose_world[P.L_HIP] + pose_world[P.R_HIP])
        up = np.asarray(sh - hip, dtype=float)
    else:
        up = CAMERA_UP.copy()
    if np.linalg.norm(up) < 1e-6 or np.linalg.norm(lat) < 1e-6:
        return None
    up = unit(up, [0.0, -1.0, 0.0])
    lat = lat - float(lat @ up) * up             # square the shoulder line to it
    if np.linalg.norm(lat) < 1e-6:
        return None
    lat = unit(lat, [1.0, 0.0, 0.0])
    fwd = np.cross(lat, up)                      # x = y cross z, right-handed
    if np.linalg.norm(fwd) < 1e-6:
        return None
    return np.column_stack([unit(fwd), lat, up])


def hand_dirs(obs: Observation, side: str, t_frame: np.ndarray):
    """(approach, across) for one hand in torso coordinates, or None.

    Prefers the 21-point hand mesh; falls back to the four hand landmarks the
    pose model carries, which are coarse but always present. Both sets use the
    same camera-aligned axes, so the same torso rotation applies to either --
    only their origins differ, and nothing here uses an origin.
    """
    hw = obs.hand_world.get(side)
    if hw is not None and len(hw) >= 21:
        approach = hw[H.MIDDLE_MCP] - hw[H.WRIST]         # down the fingers
        across = hw[H.PINKY_MCP] - hw[H.INDEX_MCP]        # across the palm
    elif obs.pose_world is not None:
        idx, pw = POSE_ARM[side], obs.pose_world
        knuckles = 0.5 * (pw[idx["index"]] + pw[idx["pinky"]])
        approach = knuckles - pw[idx["wrist"]]
        across = pw[idx["pinky"]] - pw[idx["index"]]
    else:
        return None
    if np.linalg.norm(approach) < 1e-6 or np.linalg.norm(across) < 1e-6:
        return None
    return t_frame.T @ unit(approach), t_frame.T @ unit(across)


def pinch(obs: Observation, side: str) -> float | None:
    """Thumb tip to index tip over palm length. None without a hand mesh.

    Dividing by the palm makes it independent of how far away you are, which a
    raw distance is not -- step back, and a raw threshold shuts the gripper.
    """
    hw = obs.hand_world.get(side)
    if hw is None or len(hw) < 21:
        return None
    palm = float(np.linalg.norm(hw[H.MIDDLE_MCP] - hw[H.WRIST]))
    if palm < 1e-6:
        return None
    return float(np.linalg.norm(hw[H.THUMB_TIP] - hw[H.INDEX_TIP])) / palm


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
@dataclass
class Targets:
    yaw: float = 0.0
    lift: float = 0.35
    arm: dict = field(default_factory=lambda: {s: np.zeros(7) for s in SIDES})
    grip: dict = field(default_factory=lambda: {s: 0.8 for s in SIDES})
    driving: bool = False               # are these coming from the operator?
    reason: str = "idle"                # and if not, why not
    raw: dict = field(default_factory=dict)      # pre-filter values, for the HUD


@dataclass
class Calibration:
    torso: np.ndarray                   # reference torso frame, for yaw
    shoulder_y: float                   # reference image height, for lift
    wrist: dict                         # per side, reference gripper rotation


# --------------------------------------------------------------------------
class Retargeter:
    """Calibration, filtering and joint limits wrapped around the maths above."""

    def __init__(self, cfg: TeleopConfig | None = None, limits: dict | None = None):
        self.cfg = cfg or TeleopConfig()
        self.limits = limits or DEFAULT_LIMITS
        self.cal: Calibration | None = None
        self.engaged = False
        self.crouch_span = self.cfg.crouch_span
        self._q = {s: np.zeros(7) for s in SIDES}
        # Where the last command left each channel. Rebuilding the filters --
        # which calibration does -- must not lose this, or the first command
        # afterwards is an unlimited step from wherever the robot is to
        # wherever you now are.
        self._last_yaw = 0.0
        self._last_lift = float(self.cfg.lift_top)
        self._last_grip = {s: 0.8 for s in SIDES}
        self._build_filters()

    def _build_filters(self) -> None:
        c = self.cfg
        self._f_dir = {s: {k: OneEuro(c.mincutoff_dir, c.beta_dir, shape=(3,))
                           for k in ("upper", "fore")} for s in SIDES}
        self._f_hand = {s: {k: OneEuro(c.mincutoff_hand, c.beta_hand, shape=(3,))
                            for k in ("approach", "across")} for s in SIDES}
        self._f_arm = {s: OneEuro(c.mincutoff_joint, c.beta_joint, shape=(7,))
                       for s in SIDES}
        self._r_arm = {s: RateLimit(c.joint_rate, shape=(7,)) for s in SIDES}
        self._f_grip = {s: OneEuro(c.mincutoff_grip, c.beta_grip) for s in SIDES}
        self._r_grip = {s: RateLimit(c.grip_rate) for s in SIDES}
        self._f_yaw = AngleOneEuro(c.mincutoff_base, c.beta_base)
        self._r_yaw = RateLimit(c.yaw_rate)
        self._f_lift = OneEuro(c.mincutoff_base, c.beta_base)
        self._r_lift = RateLimit(c.lift_rate)
        # Seed every rate limiter with where its channel already is, so a
        # rebuild slews from there instead of jumping.
        for s in SIDES:
            self._r_arm[s].reset(self._q[s])
            self._r_grip[s].reset(self._last_grip[s])
        self._r_yaw.reset(self._last_yaw)
        self._r_lift.reset(self._last_lift)

    # ---- calibration ------------------------------------------------------
    def calibrate(self, obs: Observation) -> bool:
        """Freeze the pose you are in now as the robot's neutral.

        Stand square to the camera with your arms down. From here on yaw is how
        far you have turned from this, lift is how far your shoulders have
        dropped below it, and (with wrist_relative) the wrist joints read zero
        here instead of at whatever angle your resting hand happens to make.
        """
        if not obs.visible(self.cfg.min_visibility):
            return False
        t_frame = torso_frame(obs.pose_world,
                               obs.hips_visible(self.cfg.min_visibility))
        if t_frame is None:
            return False

        wrist_ref = {}
        for side in SIDES:
            dirs = hand_dirs(obs, side, t_frame)
            wrist_ref[side] = None if dirs is None else gripper_frame(*dirs)

        self.cal = Calibration(
            torso=t_frame,
            shoulder_y=float(0.5 * (obs.pose_image[P.L_SHOULDER][1]
                                    + obs.pose_image[P.R_SHOULDER][1])),
            wrist=wrist_ref)
        self.reset_filters()
        if self.cfg.engage_on_calibrate:
            self.engaged = True
        return True

    def set_crouch_bottom(self, obs: Observation) -> bool:
        """Measure your crouch instead of assuming it: call this at the bottom.

        Sets the span so that exactly this much shoulder drop reaches the bottom
        of the carriage's travel.
        """
        if self.cal is None or not obs.visible(self.cfg.min_visibility):
            return False
        y = float(0.5 * (obs.pose_image[P.L_SHOULDER][1]
                         + obs.pose_image[P.R_SHOULDER][1]))
        span = y - self.cal.shoulder_y
        if span < 0.04:                   # you are not actually crouched
            return False
        self.crouch_span = span
        return True

    def reset_filters(self, hold: bool = True) -> None:
        """Rebuild the smoothing, optionally keeping where the commands are.

        `hold` is the default because the robot does not teleport when you
        recalibrate: the arms are where they are, and the new commands have to
        slew from there. Pass False only when the robot really has been put back
        to its home pose.
        """
        if not hold:
            self._q = {s: np.zeros(7) for s in SIDES}
            self._last_yaw = 0.0
            self._last_lift = float(self.cfg.lift_top)
            self._last_grip = {s: 0.8 for s in SIDES}
        self._build_filters()

    # ---- the per-frame mapping -------------------------------------------
    def __call__(self, obs: Observation) -> Targets:
        cfg = self.cfg
        # Rest is the TOP of the travel, not the middle. Standing is the pose
        # the operator calibrates in and it maps to the top, so parking
        # anywhere else means the carriage lurches the moment you engage.
        out = Targets(lift=cfg.lift_top)

        if not self.engaged:
            out.reason = "press SPACE to engage"
            return self._hold(out, obs.t)
        if self.cal is None:
            out.reason = "press C to calibrate"
            return self._hold(out, obs.t)
        if not obs.visible(cfg.min_visibility):
            out.reason = "torso not visible"
            return self._hold(out, obs.t)
        t_frame = torso_frame(obs.pose_world,
                               obs.hips_visible(self.cfg.min_visibility))
        if t_frame is None:
            out.reason = "degenerate torso"
            return self._hold(out, obs.t)

        out.driving, out.reason = True, "driving"
        pw = obs.pose_world

        # ---- tower yaw: how far you have turned since calibration ---------
        # Expressing the current torso frame in the calibrated one reduces the
        # whole thing to one rotation about your own up axis.
        rel = self.cal.torso.T @ t_frame
        psi = math.atan2(rel[1, 0], rel[0, 0])
        if abs(psi) < cfg.yaw_deadband:
            psi = 0.0
        else:
            psi -= math.copysign(cfg.yaw_deadband, psi)
        yaw_raw = float(np.clip(cfg.yaw_gain * psi, -cfg.yaw_limit, cfg.yaw_limit))

        # ---- tower lift: how far your shoulders dropped down the frame ----
        # Image space, not world space: world landmarks are hip-centred, so they
        # travel with you and cannot tell standing from crouching. The picture
        # can -- crouch, and your shoulders move down the frame.
        sh_y = float(0.5 * (obs.pose_image[P.L_SHOULDER][1]
                            + obs.pose_image[P.R_SHOULDER][1]))
        drop = (sh_y - self.cal.shoulder_y) / max(self.crouch_span, 1e-3)
        span = cfg.lift_top - cfg.lift_bottom
        lift_raw = float(cfg.lift_top - float(np.clip(drop, 0.0, 1.0)) * span)

        out.raw = dict(yaw=yaw_raw, lift=lift_raw, drop=drop, psi=psi,
                       pinch={}, arm={})

        # ---- arms ---------------------------------------------------------
        for robot_side in SIDES:
            human_side = self._human_side(robot_side)
            idx = POSE_ARM[human_side]
            upper = t_frame.T @ unit(pw[idx["elbow"]] - pw[idx["shoulder"]])
            fore = t_frame.T @ unit(pw[idx["wrist"]] - pw[idx["elbow"]])
            # Filter the directions, then renormalise: averaging unit vectors
            # shortens them, and a short "unit" vector quietly biases the asin
            # that recovers shoulder abduction.
            upper = unit(self._f_dir[robot_side]["upper"](upper, obs.t), upper)
            fore = unit(self._f_dir[robot_side]["fore"](fore, obs.t), fore)

            r_grip = None
            if cfg.use_wrist:
                dirs = hand_dirs(obs, human_side, t_frame)
                if dirs is not None:
                    appr, across = self._mirror_pair(*dirs)
                    appr = unit(self._f_hand[robot_side]["approach"](appr, obs.t), appr)
                    across = unit(self._f_hand[robot_side]["across"](across, obs.t), across)
                    r_grip = gripper_frame(appr, across)

            upper, fore = self._mirror_pair(upper, fore)

            q = solve_arm(upper, fore, r_grip, robot_side, self._q[robot_side])
            q = clamp_to_limits(q, self.limits[robot_side])
            out.raw["arm"][robot_side] = q.copy()

            q = self._f_arm[robot_side](q, obs.t)
            q = clamp_to_limits(q, self.limits[robot_side])
            q = self._r_arm[robot_side](q, obs.t)
            self._q[robot_side] = q
            out.arm[robot_side] = q

            # ---- gripper --------------------------------------------------
            p = pinch(obs, human_side)
            out.raw["pinch"][robot_side] = p
            if p is None:
                target = 0.8                     # no fingers tracked: hold open
            else:
                target = float(np.clip(
                    (p - cfg.pinch_closed)
                    / max(cfg.pinch_open - cfg.pinch_closed, 1e-6), 0.0, 1.0))
                target = float(np.clip(self._f_grip[robot_side](target, obs.t),
                                       0.0, 1.0))
            out.grip[robot_side] = float(self._r_grip[robot_side](target, obs.t))

        out.yaw = float(self._r_yaw(self._f_yaw(yaw_raw, obs.t), obs.t))
        out.lift = float(self._r_lift(self._f_lift(lift_raw, obs.t), obs.t))
        self._remember(out)
        return out

    def _remember(self, out: Targets) -> None:
        self._last_yaw, self._last_lift = out.yaw, out.lift
        for s in SIDES:
            self._last_grip[s] = out.grip[s]

    def _human_side(self, robot_side: str) -> str:
        if not self.cfg.mirror:
            return robot_side
        return "right" if robot_side == "left" else "left"

    def _mirror_pair(self, a, b):
        """Reflect two torso-frame directions through the sagittal plane.

        Mirror mode feeds your right arm to the robot's left, so the geometry
        has to be reflected too -- otherwise the robot's left arm reproduces a
        right arm's pose and the elbow points the wrong way. The sagittal plane
        is normal to y in both frames, so the reflection is one sign flip.
        """
        if not self.cfg.mirror:
            return a, b
        flip = np.array([1.0, -1.0, 1.0])
        return a * flip, b * flip

    def _hold(self, out: Targets, t: float) -> Targets:
        """Not driving: ease everything back to the robot's rest pose.

        Through the same rate limiters a live command uses, so disengaging
        relaxes the arms down instead of dropping them.
        """
        for s in SIDES:
            q = self._r_arm[s](np.zeros(7), t)
            self._q[s] = q
            out.arm[s] = q
            out.grip[s] = float(self._r_grip[s](0.8, t))
        out.yaw = float(self._r_yaw(0.0, t))
        out.lift = float(self._r_lift(out.lift, t))
        self._remember(out)
        return out
