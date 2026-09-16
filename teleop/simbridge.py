"""The MuJoCo side: build the robot, push targets into it, render it.

Nothing about teleoperation lives here and nothing about MuJoCo lives outside
here, so the retargeting can be tested without a simulator and the simulator can
be driven without a camera.

The model is the one the rest of the project already uses -- `bench.build_bench()`
joins tower.xml and the OpenArm onto an empty floor, through the same
`robot.assemble()` path the full aisle goes through. `--scene aisle` swaps in
out/scene.xml with the shelves and stock, but the default is the robot alone,
which is what you want while you are checking that the mapping is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import bench                              # noqa: E402
import robot as R                         # noqa: E402

from .config import TeleopConfig          # noqa: E402
from .landmarks import SIDES              # noqa: E402


def _make_fast(model) -> None:
    """Drop the expensive display options; see docs/VIEWER.md for the numbers."""
    model.vis.quality.shadowsize = 0
    model.vis.quality.offsamples = 0
    model.light_castshadow[:] = 0
    model.mat_reflectance[:] = 0


class SimBridge:
    """Owns the robot, the physics clock and the offscreen camera."""

    def __init__(self, cfg: TeleopConfig | None = None, pretty: bool = False):
        self.cfg = cfg or TeleopConfig()
        if self.cfg.scene == "aisle":
            self.bot = R.build(REPO / "out" / "scene.xml")
        else:
            self.bot = bench.build_bench()
        self.model, self.data = self.bot.model, self.bot.data
        if not pretty:
            _make_fast(self.model)

        # Offscreen buffer size is fixed at compile time, so a bigger request
        # than the MJCF allows would fail at render, not here. Clamp instead.
        self.rw = min(self.cfg.render_width, int(self.model.vis.global_.offwidth))
        self.rh = min(self.cfg.render_height, int(self.model.vis.global_.offheight))
        self.renderer = mujoco.Renderer(self.model, height=self.rh, width=self.rw)

        # Over the robot's shoulder, looking the same way it looks.
        #
        # The robot faces world +y, and MuJoCo's azimuth is the direction the
        # camera looks, so 90 puts the camera behind it. That is the one view
        # where the screen agrees with the mapping: the robot's left arm is the
        # arm on your left, and a reach forward goes away into the screen, the
        # same way your own does. 270 would put you face to face with it, which
        # looks friendlier and silently mirrors left and right.
        #
        # The cost is that you see its back, so the grippers are partly hidden
        # on a forward reach -- '[' and ']' swing round when you need to look.
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 2.4
        self.cam.azimuth = 90.0
        self.cam.elevation = -10.0
        self.cam.lookat[:] = [0.0, 0.19, 0.94]
        # Deliberately NOT following the carriage. Tracking it would keep the
        # arms centred, but standing up and crouching would then look identical
        # -- the lift is half of what there is to see, and it is only visible
        # against a fixed frame. 'F' turns following on to inspect the hands.
        self.follow = False
        self._fixed_lookat = np.array(self.cam.lookat)

        self.viewer = None
        self.home()

    # ---- limits, straight from the compiled model -------------------------
    @property
    def limits(self) -> dict:
        """Per-side (7, 2) arrays of the actuator ctrlrange.

        Read off the model rather than copied into the retargeter, so the
        clamping can never disagree with the robot it is clamping for.
        """
        out = {}
        for side in SIDES:
            out[side] = np.array(
                [self.model.actuator_ctrlrange[a] for a in self.bot.arm_act[side]],
                dtype=float)
        return out

    # ---- commands ---------------------------------------------------------
    def home(self) -> None:
        """Rest pose, with the carriage at the TOP of its travel.

        `robot.home()` parks the lift mid-travel, which is right for the robot's
        own scripts but wrong here: standing is the operator's neutral and it
        maps to the top, so starting mid-travel means the carriage drops the
        instant you engage, before you have moved at all.
        """
        self.bot.home()
        top = self.cfg.lift_top
        self.data.qpos[self.model.jnt_qposadr[self.bot.lift_jnt]] = top
        self.bot.set_base(0.0, top)
        mujoco.mj_forward(self.model, self.data)

    def apply(self, targets) -> None:
        """Send one set of targets to the position actuators.

        Only ctrl is written -- never qpos -- so the arms are driven by the same
        actuators a real controller would use and you see the real tracking lag,
        the real sag under gravity and the real contact response. Teleporting
        qpos would look perfect and prove nothing.
        """
        self.bot.set_base(targets.yaw, targets.lift)
        for side in SIDES:
            self.bot.set_arm(side, targets.arm[side])
            self.bot.set_gripper(side, targets.grip[side])

    def step(self, dt: float) -> int:
        n = max(1, int(round(dt / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        return n

    # ---- output -----------------------------------------------------------
    def render(self, camera: str | int | None = None) -> np.ndarray:
        if camera is None:
            if self.follow:
                # Sit a little below the shoulder mount: the arms hang from it,
                # so centring on the mount itself puts them in the bottom half.
                mount = self.bot.mount()
                self.cam.lookat[:] = [mount[0], mount[1], mount[2] - 0.20]
            else:
                self.cam.lookat[:] = self._fixed_lookat
            self.renderer.update_scene(self.data, camera=self.cam)
        else:
            self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def orbit(self, d_azimuth: float = 0.0, d_elevation: float = 0.0,
              d_distance: float = 0.0) -> None:
        self.cam.azimuth = (self.cam.azimuth + d_azimuth) % 360.0
        self.cam.elevation = float(np.clip(self.cam.elevation + d_elevation, -89.0, 89.0))
        self.cam.distance = float(np.clip(self.cam.distance + d_distance, 0.6, 8.0))

    def open_viewer(self) -> None:
        """Also open the interactive MuJoCo window, alongside the embedded view."""
        import mujoco.viewer as mj_viewer
        self.viewer = mj_viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False)

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer.sync()
        return True

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass

    # ---- state, for the HUD ----------------------------------------------
    def state(self) -> dict:
        yaw, lift = self.bot.get_base()
        return dict(
            yaw=yaw, lift=lift,
            arm={s: self.bot.get_arm(s) for s in SIDES},
            hand={s: self.bot.hand(s) for s in SIDES},
            ncon=int(self.data.ncon),
        )
