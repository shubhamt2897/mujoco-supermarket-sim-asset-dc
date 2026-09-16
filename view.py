"""Open the supermarket scene in the MuJoCo viewer.

Launching a MuJoCo sim is always the same three steps:

    1. MjModel  -- the compiled scene. Constant: geometry, masses, actuators.
    2. MjData   -- the changing state. Positions, velocities, contacts.
    3. a viewer -- a window that draws MjData against MjModel.

Everything below is those three steps plus command-line switches.

    python view.py                 scene only, no side panels, fast rendering
    python view.py --robot         same, with the bimanual arm attached
    python view.py --ui            put the control panels back
    python view.py --pretty        full shadows and anti-aliasing (slow)
    python view.py --passive       drive the physics loop yourself
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

REPO = Path(__file__).resolve().parent


# ---------------------------------------------------------------- step 1 + 2
def load(scene_path: Path, with_robot: bool):
    """Build the MjModel (scene) and the MjData (state) that goes with it."""
    if with_robot:
        # robot.py joins the scene and the arm in memory, then compiles the
        # pair into a single MjModel. Nothing on disk is touched.
        import robot
        spec = robot.RobotSpec()
        model = robot.attach_arms(scene_path, spec).compile()
        data = mujoco.MjData(model)
        robot.apply_home(model, data, spec)      # put the arms at their home pose
    else:
        # from_xml_path compiles the MJCF file into an MjModel.
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)           # fill in positions/contacts once
    return model, data


def make_fast(model: mujoco.MjModel) -> None:
    """Turn off the expensive display options.

    These are display-only fields on the already-compiled model, so changing
    them affects this window and nothing else -- out/scene.xml keeps its full
    quality settings for the dataset renders.

    Measured at 1280x720: 3469 ms/frame as generated, 363 ms/frame with all
    four of these off. Shadows are by far the biggest cost.
    """
    model.vis.quality.shadowsize = 0     # no shadow map
    model.vis.quality.offsamples = 0     # no anti-aliasing
    model.light_castshadow[:] = 0        # no light casts a shadow
    model.mat_reflectance[:] = 0         # no mirror-like floor


# ------------------------------------------------------------------- step 3
def run_managed(model, data, show_ui: bool) -> None:
    """The easy viewer: MuJoCo runs the physics loop for you, and blocks here.

    Good for looking around. You do not control the stepping.
    """
    mujoco.viewer.launch(model, data, show_left_ui=show_ui, show_right_ui=show_ui)


def run_passive(model, data, show_ui: bool) -> None:
    """The viewer you drive yourself.

    launch_passive returns immediately and hands back a handle. *You* call
    mj_step, then viewer.sync() to push the new state to the window. This is
    the shape every control loop takes -- it is where a policy, an IK solve or
    a data recorder would go later.
    """
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=show_ui, show_right_ui=show_ui
    ) as viewer:
        while viewer.is_running():
            tic = time.perf_counter()

            mujoco.mj_step(model, data)      # advance physics by one timestep
            viewer.sync()                    # show the new state

            # sleep off the remainder so the sim runs at wall-clock speed
            ahead = model.opt.timestep - (time.perf_counter() - tic)
            if ahead > 0:
                time.sleep(ahead)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, default=REPO / "out" / "scene.xml")
    ap.add_argument("--robot", action="store_true", help="attach the bimanual arm")
    ap.add_argument("--ui", action="store_true", help="show the side control panels")
    ap.add_argument("--pretty", action="store_true", help="keep shadows and AA (slow)")
    ap.add_argument("--passive", action="store_true", help="drive the loop yourself")
    args = ap.parse_args()

    model, data = load(args.scene, args.robot)
    if not args.pretty:
        make_fast(model)

    print(f"scene    : {args.scene}")
    print(f"robot    : {'attached' if args.robot else 'no'}")
    print(f"bodies   : {model.nbody}   geoms: {model.ngeom}   dof: {model.nv}")
    print(f"rendering: {'full quality' if args.pretty else 'fast (no shadows/AA)'}")
    print(f"panels   : {'on' if args.ui else 'off -- press Tab to toggle'}")
    print("\nclose the window to exit")

    if args.passive:
        run_passive(model, data, args.ui)
    else:
        run_managed(model, data, args.ui)


if __name__ == "__main__":
    main()
