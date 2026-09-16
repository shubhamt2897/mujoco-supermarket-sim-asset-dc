"""Whole-body teleoperation of the lift tower and its bimanual arm.

    python -m teleop                 webcam -> robot, one window
    python -m teleop.selftest        check the maths, no camera needed

See teleop/README.md for the mapping and the controls.
"""

from .config import TeleopConfig
from .landmarks import Observation
from .retarget import Retargeter, Targets

__all__ = ["TeleopConfig", "Observation", "Retargeter", "Targets"]
