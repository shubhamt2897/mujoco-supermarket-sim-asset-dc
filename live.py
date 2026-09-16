"""Run the scene with the robot, plus a live window per on-board camera.

Four windows: the MuJoCo viewer you drive, and one small window each for
`tower_eye`, `camera_wrist_left` and `camera_wrist_right` -- the three cameras
that ride the robot, which is what a data-collection policy would actually see.

Tkinter owns the main loop; physics is stepped from a timer callback and the
passive viewer is synced from the same place. Cameras are re-rendered one per
tick, round-robin, because rendering three every frame costs more than the
physics does.

    python live.py                  scene + robot, 4 windows
    python live.py --bench          tower and arms only, no shelves
    python live.py --width 480      camera window width (default 640, 4:3)
    python live.py --pretty         full shadows and AA (much slower)
"""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageTk

import robot as R

REPO = Path(__file__).resolve().parent

# The cameras carried by the robot itself, in the order the windows open.
ONBOARD = ("tower_eye", "camera_wrist_left", "camera_wrist_right")


def make_fast(model) -> None:
    """Drop the expensive display options; see docs/VIEWER.md for the numbers."""
    model.vis.quality.shadowsize = 0
    model.vis.quality.offsamples = 0
    model.light_castshadow[:] = 0
    model.mat_reflectance[:] = 0


class CameraWindow:
    """One always-on-top window showing a single camera."""

    def __init__(self, master, title: str, width: int, height: int):
        self.win = tk.Toplevel(master)
        self.win.title(title)
        self.win.resizable(False, False)
        self.label = tk.Label(self.win, bd=0)
        self.label.pack()
        self.caption = tk.Label(self.win, text=title, anchor="w",
                                font=("Consolas", 9), fg="#ddd", bg="#222")
        self.caption.pack(fill="x")
        self.win.configure(bg="#222")
        self._photo = None
        blank = Image.new("RGB", (width, height), (30, 30, 34))
        self.show(np.asarray(blank))

    def show(self, rgb: np.ndarray) -> None:
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=self._photo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true", help="tower and arms only")
    ap.add_argument("--width", type=int, default=640,
                    help="camera window width; height follows 4:3 (default 640x480)")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    if args.bench:
        import bench
        bot = bench.build_bench()
    else:
        bot = R.build()
    model, data = bot.model, bot.data
    if not args.pretty:
        make_fast(model)

    cams = [c for c in ONBOARD
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) >= 0]
    missing = [c for c in ONBOARD if c not in cams]
    if missing:
        raise KeyError(f"camera(s) not in the compiled model: {missing}")

    w = args.width
    h = int(w * 3 / 4)          # 4:3, matching the 640x480 sensor
    print(f"scene    : {'bench (tower + arms)' if args.bench else 'full aisle'}")
    print(f"bodies   : {model.nbody}   geoms: {model.ngeom}   actuators: {model.nu}")
    print(f"windows  : MuJoCo viewer + {len(cams)} camera windows at {w}x{h}")
    for c in cams:
        print(f"           {c}")
    print("\nclose the MuJoCo viewer to exit")

    root = tk.Tk()
    root.title("robot cameras")
    root.geometry("260x90")
    status = tk.Label(root, text="starting...", font=("Consolas", 9), justify="left")
    status.pack(padx=8, pady=8, anchor="w")

    windows = {c: CameraWindow(root, c, w, h) for c in cams}
    renderer = mujoco.Renderer(model, height=h, width=w)
    viewer = mujoco.viewer.launch_passive(model, data,
                                          show_left_ui=False, show_right_ui=False)

    state = {"turn": 0, "steps": 0, "t0": time.perf_counter()}
    steps_per_tick = max(1, int(0.033 / model.opt.timestep))

    def tick():
        if not viewer.is_running():
            renderer.close()
            viewer.close()
            root.destroy()
            return
        for _ in range(steps_per_tick):
            mujoco.mj_step(model, data)
        state["steps"] += steps_per_tick
        viewer.sync()

        # one camera per tick keeps the loop responsive
        cam = cams[state["turn"] % len(cams)]
        state["turn"] += 1
        renderer.update_scene(data, camera=cam)
        windows[cam].show(renderer.render())

        dt = time.perf_counter() - state["t0"]
        if dt > 1.0:
            sps = state["steps"] / dt
            yaw, lift = bot.get_base()
            status.configure(
                text=f"{sps:6.0f} steps/s = {sps*model.opt.timestep:.2f}x RT\n"
                     f"yaw  {np.degrees(yaw):+7.1f} deg\n"
                     f"lift {lift:7.3f} m")
            state["steps"] = 0
            state["t0"] = time.perf_counter()
        root.after(1, tick)

    root.after(50, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
