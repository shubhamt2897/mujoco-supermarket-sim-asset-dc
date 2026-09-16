"""Landmark indices and the observation record the rest of the pipeline reads.

MediaPipe names landmarks from the *subject's* point of view: LEFT_SHOULDER is
the shoulder on your left arm, which in an un-mirrored webcam image appears on
the right of the picture. The robot's arms are named the same way -- `left` is
the arm on the robot's own left -- so subject-left driving robot-left is the
mapping that makes you feel inside the robot rather than facing it. `--mirror`
on the app swaps that if you would rather work against a reflection.

Two coordinate sets come out of MediaPipe and they are not interchangeable:

  image  (x, y) normalised to [0, 1] across the frame, y down from the top.
         Anchored to the picture, so it is the one that knows you crouched --
         your shoulders move *down the frame*. Drives the lift.

  world  metres, right-handed, axes aligned with the image: +x is image-right,
         +y is image-down, +z is away from the camera. The origin is the
         midpoint of the hips for pose, and the hand's own centre for hands,
         so only *differences* are comparable between the two sets -- which is
         all the retargeter ever takes. Drives everything angular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- pose (33)
class P:
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_PINKY, R_PINKY = 17, 18
    L_INDEX, R_INDEX = 19, 20
    L_THUMB, R_THUMB = 21, 22
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28


# Per side, the landmarks the retargeter needs. Indexing a dict by "left" /
# "right" keeps every downstream loop side-agnostic.
POSE_ARM = {
    "left":  dict(shoulder=P.L_SHOULDER, elbow=P.L_ELBOW, wrist=P.L_WRIST,
                  index=P.L_INDEX, pinky=P.L_PINKY, thumb=P.L_THUMB),
    "right": dict(shoulder=P.R_SHOULDER, elbow=P.R_ELBOW, wrist=P.R_WRIST,
                  index=P.R_INDEX, pinky=P.R_PINKY, thumb=P.R_THUMB),
}

# The landmarks that must be visible before anything is allowed to drive the
# robot: just the two shoulders. Everything angular is measured relative to the
# shoulder line, and that is the one thing the mapping genuinely cannot do
# without.
#
# The hips are NOT required, though they are used when they are there. Sitting
# at a desk -- which is how most of this gets tested -- puts them out of frame or
# behind the desk, and MediaPipe reports them at ~0.3 visibility. Demanding them
# meant the robot simply never moved for a seated operator. See
# `retarget.torso_frame` for what stands in for them.
REQUIRED = (P.L_SHOULDER, P.R_SHOULDER)
HIPS = (P.L_HIP, P.R_HIP)


# ---------------------------------------------------------------- hand (21)
class H:
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


SIDES = ("left", "right")


# ------------------------------------------------------------------ record
@dataclass
class Observation:
    """One tracked frame, in the form the retargeter wants it.

    Everything is plain numpy so the retargeter never imports mediapipe -- that
    is what lets `synthetic.py` fabricate observations and `selftest.py` run the
    whole pipeline with no camera and no model files present.
    """

    t: float                                   # seconds, monotonic
    pose_world: np.ndarray | None = None       # (33, 3) metres
    pose_image: np.ndarray | None = None       # (33, 3) normalised
    pose_vis: np.ndarray | None = None         # (33,) visibility 0..1
    hand_world: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {"left": None, "right": None})   # (21, 3) each
    hand_image: dict[str, np.ndarray | None] = field(
        default_factory=lambda: {"left": None, "right": None})
    frame: np.ndarray | None = None            # BGR, for the UI only
    latency_ms: float = 0.0

    @property
    def has_pose(self) -> bool:
        return self.pose_world is not None and self.pose_image is not None

    def visible(self, min_vis: float = 0.5) -> bool:
        """True when the torso the whole mapping is built on is actually seen."""
        if not self.has_pose:
            return False
        if self.pose_vis is None:
            return True
        return bool(np.all(self.pose_vis[list(REQUIRED)] >= min_vis))

    def hips_visible(self, min_vis: float = 0.5) -> bool:
        """Whether the hips can be trusted to set the torso's up axis."""
        if not self.has_pose:
            return False
        if self.pose_vis is None:
            return True
        return bool(np.all(self.pose_vis[list(HIPS)] >= min_vis))


def pose_connections() -> list[tuple[int, int]]:
    """Skeleton edges for drawing, from MediaPipe if present.

    mediapipe 1.0 dropped `mp.solutions.drawing_utils`, so the overlay draws the
    skeleton itself. The connection tables survived the move into
    `tasks.vision`; the literal fallback keeps the UI working if that moves too.
    """
    try:
        from mediapipe.tasks.python.vision import PoseLandmarksConnections as C
        return [(c.start, c.end) for c in C.POSE_LANDMARKS]
    except Exception:
        return [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28), (15, 17), (15, 19), (16, 18), (16, 20)]


def hand_connections() -> list[tuple[int, int]]:
    try:
        from mediapipe.tasks.python.vision import HandLandmarksConnections as C
        return [(c.start, c.end) for c in C.HAND_CONNECTIONS]
    except Exception:
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
                (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)]
