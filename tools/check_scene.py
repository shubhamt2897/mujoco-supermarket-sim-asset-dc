"""Compile a scene, settle it, and report whether it is sane and fast."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


def check(xml_path: Path, seconds: float = 3.0, verbose: bool = True):
    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    t_compile = time.time() - t0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    free_bodies = [i for i in range(model.nbody)
                   if model.body_jntnum[i] == 1
                   and model.jnt_type[model.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE]
    start = np.array([data.xpos[i].copy() for i in free_bodies])

    n_steps = int(seconds / model.opt.timestep)
    t0 = time.time()
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    wall = time.time() - t0

    end = np.array([data.xpos[i].copy() for i in free_bodies])
    drift = np.linalg.norm(end - start, axis=1) * 1000.0
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    sps = n_steps / wall

    if verbose:
        print(f"model            : {xml_path}")
        print(f"  compile time   : {t_compile:6.2f} s")
        print(f"  nbody / ngeom  : {model.nbody} / {model.ngeom}")
        print(f"  nmesh / nq,nv  : {model.nmesh} / {model.nq},{model.nv}")
        print(f"  free bodies    : {len(free_bodies)}")
        print(f"  total mass     : {mujoco.mj_getTotalmass(model):.2f} kg")
        print(f"  --- after {seconds:.0f} s of physics ---")
        print(f"  contacts       : {data.ncon}")
        print(f"  all finite     : {finite}")
        print(f"  drift mean/max : {drift.mean():.2f} / {drift.max():.2f} mm")
        print(f"  over 2 mm      : {(drift > 2).sum()} of {len(drift)}")
        print(f"  speed          : {sps:.0f} steps/s = {sps*model.opt.timestep:.2f}x realtime")
    return dict(model=model, data=data, drift=drift, finite=finite, sps=sps,
                ncon=int(data.ncon), free_bodies=free_bodies, compile_s=t_compile)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", type=Path, nargs="?", default=Path("out/scene.xml"))
    ap.add_argument("--seconds", type=float, default=3.0)
    a = ap.parse_args()
    check(a.xml, a.seconds)
