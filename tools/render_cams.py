"""Render every named camera to out/shots/ after letting the scene settle."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def render_all(xml: Path, out_dir: Path, seconds: float = 3.0,
               width: int = 640, height: int = 480):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

    cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for cam in cams:
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            path = out_dir / f"{cam}.png"
            imageio.imwrite(path, img)
            written.append((cam, path, float(img.mean())))
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", type=Path, nargs="?", default=Path("out/scene.xml"))
    ap.add_argument("--out", type=Path, default=Path("out/shots"))
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    for cam, path, mean in render_all(a.xml, a.out, a.seconds):
        print(f"  {cam:14s} -> {path}   mean pixel {mean:6.1f}")
