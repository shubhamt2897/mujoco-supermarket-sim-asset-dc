"""Every tunable in one place, so the mapping can be adjusted without reading
the maths.

The defaults were chosen for a laptop webcam at arm's length with your whole
upper body in frame. If the robot feels sluggish, raise `beta_*`; if it
twitches at rest, lower `mincutoff_*`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TeleopConfig:
    # ---- tracking ---------------------------------------------------------
    camera: int = 0
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    min_visibility: float = 0.5       # torso landmarks below this = no drive
    tracker: str = "holistic"         # "holistic" (adds fingers) or "pose"

    # Auto-exposure is the biggest single source of teleop lag: in dim light
    # the webcam lengthens its exposure to brighten the picture, and a longer
    # exposure means a longer frame. Measured here, cap.read() went from 33 ms
    # locked to 71 ms on auto -- half the tracking rate, lost before a single
    # landmark had been computed.
    #
    # It is still the default, deliberately. Locking picks one exposure at
    # startup, and the light in a room does not hold still: a value that works
    # this morning is wrong by the afternoon, and a teleop rig that has to be
    # relaunched when the sun moves is worse than a slow one. Opt in with
    # --lock-exposure when the lighting is under control and the frame rate
    # matters more.
    #
    # `min_luma` is the brightness below which the lock is loosened a stop and
    # then abandoned, because a picture too dark to track is worse than both.
    lock_exposure: bool = False
    min_luma: float = 55.0            # mean 0..255 over the frame

    # ---- tower lift: standing high, crouching low -------------------------
    # Driven by how far your shoulders travel DOWN the image, as a fraction of
    # frame height, measured from wherever you were standing when you calibrated.
    # 0.22 means a crouch that drops your shoulders through 22% of the frame
    # takes the carriage through its whole 0.70 m. Recalibrate with 'v' at the
    # bottom of a squat and this is measured off you instead of guessed.
    crouch_span: float = 0.22
    lift_top: float = 0.70            # standing  -> top of travel
    lift_bottom: float = 0.0          # crouched  -> bottom of travel
    lift_rate: float = 0.6            # m/s slew limit on the carriage

    # ---- tower yaw: turn your shoulders, the tower slews ------------------
    # Torso yaw is recovered from MediaPipe's world-landmark z, which is its
    # weakest axis, so it gets a deadband to sit still when you are square to
    # the camera and a gain so a comfortable turn reaches a useful angle.
    yaw_gain: float = 1.6
    yaw_deadband: float = math.radians(6.0)
    yaw_limit: float = math.pi
    yaw_rate: float = 2.0             # rad/s slew limit

    # ---- gripper: pinch to close -----------------------------------------
    # Thumb tip to index tip, over the length of the palm, so it is invariant to
    # how far from the camera your hand is.
    pinch_closed: float = 0.35        # at or below -> jaws shut
    pinch_open: float = 1.25          # at or above -> jaws wide
    grip_rate: float = 4.0            # 1/s slew limit on the open fraction

    # ---- wrist ------------------------------------------------------------
    # With wrist_relative on, the calibration pose is defined to be the robot's
    # zero wrist, and only the change since then is sent. Without it, your
    # hand's absolute orientation is matched, which is correct but parks the
    # wrist joints at an awkward angle the moment you stand naturally.
    wrist_relative: bool = True
    use_wrist: bool = True            # off = wrist joints held at zero

    # ---- smoothing --------------------------------------------------------
    # Directions are filtered BEFORE the angles are solved, not after.
    # Smoothing a joint angle means smoothing the output of a nonlinear map,
    # where a couple of noisy landmarks can already have swung the angle tens of
    # degrees; smoothing the unit vectors that go in is far better conditioned.
    #
    # The hand gets its own, much heavier setting. Measured over a recorded
    # session the wrist joints were the shakiest by a wide margin -- j5 9.6 deg
    # of high-frequency residual against 2-4 deg for the shoulder and elbow --
    # because the palm axis is read across a ~4 cm baseline between two knuckle
    # landmarks, so a millimetre of landmark noise is degrees of roll.
    mincutoff_dir: float = 1.2
    beta_dir: float = 0.04
    mincutoff_hand: float = 0.6
    beta_hand: float = 0.015

    mincutoff_joint: float = 1.5
    beta_joint: float = 0.10
    mincutoff_base: float = 0.8       # yaw and lift: steadier than the arms
    beta_base: float = 0.03
    mincutoff_grip: float = 2.0
    beta_grip: float = 0.10
    joint_rate: float = 6.0           # rad/s slew limit per arm joint

    # ---- behaviour --------------------------------------------------------
    mirror: bool = False              # swap which of your arms drives which
    engage_on_calibrate: bool = True  # start driving as soon as 'c' is pressed

    # ---- simulation -------------------------------------------------------
    scene: str = "bench"              # "bench" = robot only; "aisle" = full scene
    fullscreen: bool = False
    # Hands-free calibration runs in two phases, because pressing the air
    # button needs a hand UP and calibrating needs both arms DOWN -- asking for
    # both at once is a contradiction the operator cannot satisfy.
    #   ready_s : ungated. Lower your arms. The clock runs regardless.
    #   the rest: gated on the pose checks, and pauses if you drift.
    countdown_s: float = 5.0          # total hands-free calibration delay
    ready_s: float = 2.5              # of which this much is "get into position"
    # Never let a flaky check block calibration forever: past this, calibrate
    # anyway and say what was off. A slightly crooked neutral you can redo beats
    # a session where the robot never moved at all, which is what the first
    # recorded run actually produced.
    calibrate_timeout_s: float = 18.0
    sim_hz: float = 60.0              # control/UI rate; physics runs at its own
    render_width: int = 640
    render_height: int = 640
