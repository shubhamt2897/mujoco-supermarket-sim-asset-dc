"""Bring up the tower and arms on their own, with nothing else in the world.

This is a testbench for the joints: no shelves, no products, no roll cage, so
nothing can obstruct the slew and nothing distracts from what the tower is
doing. It goes through the same `robot.assemble()` path the full scene uses, so
the model you drive here is the model that ends up in the aisle.

The control panel is ON by default -- that is where the actuator sliders live,
and dragging them is the point of this script.

    python bench.py                 tower + arms, panels on, fast rendering
    python bench.py --no-ui         hide the panels
    python bench.py --pretty        full shadows and anti-aliasing
    python bench.py --sweep         run a scripted joint sweep, no window
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

import robot as R

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "out"

# A world containing only a floor, some light, and the site the tower stands on.
# `tower_base_site` is the same name scene.py emits, so robot.assemble() needs
# no special case for the bench.
BENCH_XML = """<mujoco model="tower_bench">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <headlight ambient="0.42 0.42 0.43" diffuse="0.25 0.25 0.25" specular="0.08 0.08 0.08"/>
    <quality shadowsize="2048" offsamples="4"/>
    <global offwidth="1920" offheight="1080"/>
    <map znear="0.003" zfar="5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.93 0.94 0.95" rgb2="0.82 0.84 0.86" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.82 0.82 0.81" rgb2="0.74 0.74 0.73" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" texuniform="true"
              specular="0.2" shininess="0.3"/>
  </asset>
  <worldbody>
    <light name="key" pos="1.2 -1.2 3.0" dir="-0.35 0.35 -1" directional="true"
           diffuse="0.45 0.45 0.45" castshadow="true"/>
    <light name="fill" pos="-1.5 1.5 2.5" dir="0.4 -0.4 -1" directional="true"
           diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" material="grid_mat" size="6 6 0.1"
          contype="1" conaffinity="2"/>
    <camera name="bench_front" mode="fixed" pos="0 -2.6 1.5"
            xyaxes="1 0 0  0 0.42 0.91" fovy="48"/>
    <camera name="bench_side" mode="fixed" pos="2.6 0 1.5"
            xyaxes="0 1 0  -0.42 0 0.91" fovy="48"/>
    <camera name="bench_top" mode="fixed" pos="0 0 3.6"
            xyaxes="1 0 0  0 1 0" fovy="55"/>
    <site name="tower_base_site" pos="0 0 0" size="0.010" rgba="0.95 0.4 0.1 0.35"/>
  </worldbody>
</mujoco>
"""


def build_bench(spec: R.RobotSpec | None = None) -> R.Robot:
    """Write the bare world, then attach the tower and arms onto it."""
    spec = spec or R.RobotSpec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bench_path = OUT_DIR / "bench.xml"
    bench_path.write_text(BENCH_XML, encoding="utf-8")

    model = R.assemble(bench_path, spec).compile()
    bot = R.Robot(model, mujoco.MjData(model), spec)
    bot.home()
    return bot


def make_fast(model) -> None:
    model.vis.quality.shadowsize = 0
    model.vis.quality.offsamples = 0
    model.light_castshadow[:] = 0
    model.mat_reflectance[:] = 0


def print_controls(bot: R.Robot) -> None:
    """List every actuator with its range, so the sliders are readable."""
    m = bot.model
    print(f"\n{m.nu} actuators -- these are the sliders in the Control panel:")
    print(f"  {'#':>3s} {'name':22s} {'range':>22s}   {'units':8s}")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = m.actuator_ctrlrange[i]
        if name in (bot.spec.lift_actuator,):
            units, extra = "metres", ""
        elif "finger" in (name or ""):
            units, extra = "rad", "  (0 = closed)"
        else:
            units, extra = "rad", f"  = {math.degrees(lo):+.0f}..{math.degrees(hi):+.0f} deg"
        print(f"  {i:3d} {name:22s} {lo:10.4f} .. {hi:8.4f}   {units:8s}{extra}")
    print("\n  tower_yaw_ctrl  turns the whole tower")
    print("  tower_lift_ctrl raises the carriage the arms hang from")


def sweep(bot: R.Robot) -> None:
    """Drive each base joint through its range and report tracking, no window."""
    m, d = bot.model, bot.data
    print("\nyaw sweep (lift held at 0.35 m):")
    for deg in (0, 45, 90, 180, -45, -90, -180, 0):
        bot.set_base(math.radians(deg), 0.35)
        for _ in range(4000):
            mujoco.mj_step(m, d)
        got = math.degrees(bot.get_base()[0])
        print(f"  cmd {deg:+7.1f} deg -> {got:+9.3f} deg   err {got-deg:+7.4f} deg")

    print("\nlift sweep (yaw held at 0):")
    for cmd in (0.00, 0.175, 0.35, 0.525, 0.70, 0.00):
        bot.set_base(0.0, cmd)
        for _ in range(4000):
            mujoco.mj_step(m, d)
        got = bot.get_base()[1]
        print(f"  cmd {cmd:6.3f} m   -> {got:8.4f} m    sag {(cmd-got)*1000:+6.2f} mm")

    print("\nboth at once (yaw and lift are independent):")
    for deg, lift in ((90, 0.70), (-90, 0.00), (45, 0.35)):
        bot.set_base(math.radians(deg), lift)
        for _ in range(4000):
            mujoco.mj_step(m, d)
        y, l = bot.get_base()
        print(f"  cmd ({deg:+5.0f} deg, {lift:.2f} m) -> "
              f"({math.degrees(y):+8.3f} deg, {l:6.4f} m)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-ui", action="store_true", help="hide the control panels")
    ap.add_argument("--pretty", action="store_true", help="full shadows and AA")
    ap.add_argument("--sweep", action="store_true", help="scripted joint sweep, no window")
    args = ap.parse_args()

    t0 = time.perf_counter()
    bot = build_bench()
    m, d = bot.model, bot.data
    if not args.pretty:
        make_fast(m)

    print(f"tower + arms only, assembled in {time.perf_counter()-t0:.2f} s")
    print(f"  nbody/ngeom : {m.nbody} / {m.ngeom}")
    print(f"  njnt / nu   : {m.njnt} / {m.nu}")
    yaw, lift = bot.get_base()
    print(f"  base at home: yaw {math.degrees(yaw):+.2f} deg, lift {lift:.4f} m")
    print(f"  arm_mount   : {np.round(bot.mount(), 4)}")
    for side in bot.spec.sides:
        print(f"  {side:5s} hand  : {np.round(bot.hand(side), 4)}")

    if args.sweep:
        sweep(bot)
        return

    print_controls(bot)
    print("\n  Ctrl+drag a slider to move a joint. Tab toggles the panels.")
    print("  '[' and ']' cycle the cameras. Close the window to exit.")

    import mujoco.viewer as mj_viewer
    mj_viewer.launch(m, d, show_left_ui=not args.no_ui, show_right_ui=not args.no_ui)


if __name__ == "__main__":
    main()
