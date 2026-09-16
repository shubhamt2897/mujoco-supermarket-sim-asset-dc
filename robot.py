"""Attach the lift tower and the OpenArm v2 bimanual arm to the scene.

Nothing on disk is edited. Three MJCFs -- the generated scene, the hand-authored
tower, and the arm shipped with openarm_mujoco -- are joined in memory with
MjSpec and compiled together:

    scene.xml  --(at tower_base_site)-->  tower.xml
                                            |
                                            +--(at arm_mount, +90 deg yaw)--> arm

`build()` hands back a Robot, which owns the model and data and exposes the
base (yaw, lift) and the two arms as independently commandable groups.

    python robot.py            settle and report
    python robot.py --view     open the viewer
    python robot.py --shots    write one image per camera
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "out"


@dataclass(frozen=True)
class RobotSpec:
    """Everything specific to this robot, so swapping in another one is a
    matter of editing this block and nothing else.

    Every field below was read off the shipped models, not assumed.
    """
    arm_xml: Path = field(
        default_factory=lambda:
        Path(sys.prefix) / "share" / "openarm_mujoco" / "v2" / "openarm_bimanual.xml")
    tower_xml: Path = field(default_factory=lambda: REPO / "tower.xml")

    sides: tuple[str, ...] = ("left", "right")
    joint: str = "openarm_{side}_joint{k}"
    actuator: str = "{side}_joint{k}_ctrl"
    gripper_actuator: str = "{side}_finger1_ctrl"
    ee_site: str = "{side}_ee_control_point"
    finger_bodies: tuple[str, str] = ("openarm_{side}_ee_inner_finger",
                                      "openarm_{side}_ee_outer_finger")

    tower_site: str = "tower_base_site"     # in the scene: where the tower stands
    mount_site: str = "arm_mount"           # on the tower carriage: where arms bolt
    yaw_actuator: str = "tower_yaw_ctrl"
    lift_actuator: str = "tower_lift_ctrl"
    yaw_joint: str = "tower_yaw"
    lift_joint: str = "tower_lift"
    column_body: str = "tower_column"

    n_arm_joints: int = 7
    # The robot faces +x in its own frame; the near gondola is at +y, so the
    # mount is yawed a quarter turn. Without it the hands splay across the
    # aisle instead of along the shelf.
    mount_yaw: float = math.pi / 2

    # Left and right are mirrored: one posture is stored and the right arm is
    # this pattern times it. Sending both arms the same angles looks fine and
    # is wrong. (At the qpos=0 home below the mirror is a no-op, but it is what
    # any non-zero posture must go through.)
    mirror: tuple[int, ...] = (-1, -1, -1, 1, -1, -1, -1)
    # The OpenArm's own rest configuration: both arms hang straight down from
    # the shoulders. No IK needed, and nothing to get wrong.
    home_left: tuple[float, ...] = (0.0,) * 7

    home_yaw: float = 0.0
    home_lift: float = 0.35                 # mid-travel of the 0..0.70 m lift
    gripper_open_frac: float = 0.80

    # Wrist camera override: the shipped pose sees only empty aisle.
    # Swept after the near-plane fix. The earlier pose was tuned against a
    # clipped view: a 128 mm near plane was deleting the wrist that actually
    # sits in front of the lens, so the framing only looked acceptable. With
    # the plane at 19 mm the same pose put 32% arm and 1.9% finger in frame.
    # This one measures 11.9% arm, 11.3% finger, fingers centred at row 0.74.
    wrist_cam_offset: float = 0.09      # out to the side of the gripper base
    wrist_cam_z: float = -0.06          # down towards the fingertips
    wrist_cam_tilt_deg: float = 30.0
    wrist_cam_fovy: float = 60.0

    # On-board cameras render at 480p (640x480), the VGA mode a RealSense
    # actually runs. Everything that renders these cameras takes its default
    # size from here, so there is one number to change.
    cam_width: int = 640
    cam_height: int = 480

    reach: float = 0.589
    vertical_half_envelope: float = 0.435

    def mirrored(self, home_left=None) -> dict[str, np.ndarray]:
        h = np.asarray(self.home_left if home_left is None else home_left, dtype=float)
        return {"left": h, "right": h * np.asarray(self.mirror, dtype=float)}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def _frame_at(spec: mujoco.MjSpec, site_name: str, yaw: float = 0.0):
    """A frame sitting on a named site, optionally yawed about z."""
    site = spec.site(site_name)
    frame = site.parent.add_frame()
    frame.pos = site.pos
    frame.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return frame


def assemble(scene_xml: Path, spec: RobotSpec) -> mujoco.MjSpec:
    """Join scene + tower + arm in memory, in that order.

    The tower must go on first: it is the tower that carries `arm_mount`, so
    the arm's frame cannot be located until the tower is part of the scene.

    Each model is attached whole rather than body-by-body. Attaching the two
    arm base bodies out of one spec raises "incompatible id in mesh array";
    attaching from a fresh spec per side works but re-imports the entire arm
    asset set twice (+188 meshes against +94) and buries every site name under
    a per-side prefix.
    """
    scene = mujoco.MjSpec.from_file(str(scene_xml))
    scene.attach(mujoco.MjSpec.from_file(str(spec.tower_xml)),
                 prefix="", frame=_frame_at(scene, spec.tower_site))
    scene.attach(mujoco.MjSpec.from_file(str(spec.arm_xml)),
                 prefix="", frame=_frame_at(scene, spec.mount_site, spec.mount_yaw))

    # Each gripper's two jaw bodies overlap slightly where they meet the knuckle,
    # at every opening from fully closed to fully open. On the left that contact
    # pinned the finger actuator at its +7 N.m limit and the jaw would not move
    # at all: commanded fully open it reached 0.0005 rad and the jaws stayed at
    # their 91 mm closed spread. The right side happened to escape it. Excluding
    # the pair costs nothing -- a jaw cannot usefully collide with its own other
    # half -- and it is what makes the left gripper work.
    for side in spec.sides:
        inner, outer = (name.format(side=side) for name in spec.finger_bodies)
        ex = scene.add_exclude()
        ex.name = f"{side}_finger_jaw_pair"
        ex.bodyname1 = inner
        ex.bodyname2 = outer

    _aim_wrist_cameras(scene, spec)
    return scene


def _aim_wrist_cameras(scene: mujoco.MjSpec, spec: RobotSpec) -> None:
    """Re-aim the wrist cameras the arm ships with, and give them a body.

    As shipped both looked away from the work: segmentation put 0.00% finger,
    0.00% product and 1.14% shelf in frame -- they saw empty aisle. Swept over
    offset, tilt and fovy, the pose below is the first that puts the fingers in
    the lower half of the frame (8.0% of pixels, centroid at row 0.63) with the
    approach direction down the middle.

    The arm MJCF lives in site-packages and is not edited; the override happens
    here, on the in-memory spec.
    """
    tilt = math.radians(spec.wrist_cam_tilt_deg)
    sin_t, cos_t = math.sin(tilt), math.cos(tilt)
    for side in spec.sides:
        cam = scene.camera(f"camera_wrist_{side}")
        # No left/right sign flip here. The two ee_base_link frames come out
        # with identical world rotation -- the arms are mirrored in their joint
        # angles, not in their link frames -- so mirroring the camera pose
        # de-synchronises the views instead of matching them. Measured: with a
        # sign flip the fingers filled 7.99% of the left view and 2.90% of the
        # right; without it both read 7.99% at row 0.62.
        cam.pos = [spec.wrist_cam_offset, 0.0, spec.wrist_cam_z]
        # These cameras carry their orientation in `alt` as euler angles, and
        # `alt` wins over `quat` -- assigning cam.quat alone is silently
        # ignored. Switch `alt` to xyaxes and write the axes there instead.
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [0.0, -1.0, 0.0,      # camera x (right)
                          cos_t, 0.0, -sin_t]  # camera y (up)
        cam.fovy = spec.wrist_cam_fovy
        cam.resolution = [spec.cam_width, spec.cam_height]

        # A RealSense-ish body so the camera is visible rather than an
        # invisible point. It must sit BEHIND the lens: centred on the camera
        # (as it first was) the camera is inside the box looking out through
        # backfaces, which MuJoCo does not draw.
        depth = 0.0125
        back = depth + 0.002
        body = scene.body(f"openarm_{side}_ee_base_link")
        g = body.add_geom()
        g.name = f"wrist_cam_body_{side}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [spec.wrist_cam_offset + back * sin_t,
                 0.0,
                 spec.wrist_cam_z + back * cos_t]
        g.size = [0.030, 0.011, depth]
        g.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        g.alt.xyaxes = [0.0, -1.0, 0.0, cos_t, 0.0, -sin_t]
        g.rgba = [0.13, 0.14, 0.16, 1.0]
        g.contype, g.conaffinity = 0, 0
    return scene


class Robot:
    """Handle on the compiled model: base pose and arm pose, commanded apart."""

    def __init__(self, model, data, spec: RobotSpec):
        self.model, self.data, self.spec = model, data, spec

        def act(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise KeyError(f"actuator {name!r} not in the compiled model")
            return i

        def jnt(name):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise KeyError(f"joint {name!r} not in the compiled model")
            return i

        self.yaw_act, self.lift_act = act(spec.yaw_actuator), act(spec.lift_actuator)
        self.yaw_jnt, self.lift_jnt = jnt(spec.yaw_joint), jnt(spec.lift_joint)
        self.arm_act = {s: [act(spec.actuator.format(side=s, k=k))
                            for k in range(1, spec.n_arm_joints + 1)]
                        for s in spec.sides}
        self.arm_qadr = {s: [model.jnt_qposadr[jnt(spec.joint.format(side=s, k=k))]
                             for k in range(1, spec.n_arm_joints + 1)]
                         for s in spec.sides}
        self.grip_act = {s: act(spec.gripper_actuator.format(side=s))
                         for s in spec.sides}
        self.ee = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                        spec.ee_site.format(side=s))
                   for s in spec.sides}
        self.mount_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                               spec.mount_site)

    # ---- base -------------------------------------------------------------
    def _clip(self, act_id, value):
        lo, hi = self.model.actuator_ctrlrange[act_id]
        return float(np.clip(value, lo, hi))

    def set_base(self, yaw: float, lift: float) -> None:
        """Command the tower: yaw in radians, lift in metres.

        Both are clipped to the actuator's own ctrlrange as read from the
        compiled model, so this cannot be driven past the hardware limits.
        """
        self.data.ctrl[self.yaw_act] = self._clip(self.yaw_act, yaw)
        self.data.ctrl[self.lift_act] = self._clip(self.lift_act, lift)

    def get_base(self) -> tuple[float, float]:
        """Measured (yaw, lift) -- where the tower actually is, not what it was told."""
        return (float(self.data.qpos[self.model.jnt_qposadr[self.yaw_jnt]]),
                float(self.data.qpos[self.model.jnt_qposadr[self.lift_jnt]]))

    def get_base_target(self) -> tuple[float, float]:
        return (float(self.data.ctrl[self.yaw_act]),
                float(self.data.ctrl[self.lift_act]))

    # ---- arms -------------------------------------------------------------
    def set_arm(self, side: str, q) -> None:
        self.data.ctrl[self.arm_act[side]] = np.asarray(q, dtype=float)

    def get_arm(self, side: str) -> np.ndarray:
        return self.data.qpos[self.arm_qadr[side]].copy()

    def set_gripper(self, side: str, frac: float) -> None:
        """frac 0 = closed, 1 = fully open, on whichever end of this actuator's
        own ctrlrange is further from zero."""
        lo, hi = self.model.actuator_ctrlrange[self.grip_act[side]]
        open_end = hi if abs(hi) > abs(lo) else lo
        self.data.ctrl[self.grip_act[side]] = open_end * float(np.clip(frac, 0.0, 1.0))

    # ---- poses ------------------------------------------------------------
    def home(self) -> None:
        """Arms at their natural rest, tower at mid-lift, grippers open."""
        s = self.spec
        for side, q in s.mirrored().items():
            self.data.qpos[self.arm_qadr[side]] = q
            self.data.ctrl[self.arm_act[side]] = q
            self.set_gripper(side, s.gripper_open_frac)
        self.data.qpos[self.model.jnt_qposadr[self.yaw_jnt]] = s.home_yaw
        self.data.qpos[self.model.jnt_qposadr[self.lift_jnt]] = s.home_lift
        self.set_base(s.home_yaw, s.home_lift)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds: float) -> float:
        n = int(seconds / self.model.opt.timestep)
        t0 = time.perf_counter()
        self.step(n)
        return n / (time.perf_counter() - t0)

    # ---- queries ----------------------------------------------------------
    def hand(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.ee[side]].copy()

    def mount(self) -> np.ndarray:
        return self.data.site_xpos[self.mount_site_id].copy()

    def hand_in_mount_frame(self, side: str) -> np.ndarray:
        """Hand position expressed in the tower's mount frame.

        In world coordinates the two hands mirror about the tower's own axis,
        which only lines up with world x when yaw is zero. Comparing them here
        makes the mirror test independent of where the tower is pointing.
        """
        R = self.data.site_xmat[self.mount_site_id].reshape(3, 3)
        return R.T @ (self.hand(side) - self.mount())


def build(scene_xml: Path | None = None, spec: RobotSpec | None = None) -> Robot:
    spec = spec or RobotSpec()
    scene_xml = scene_xml or (OUT_DIR / "scene.xml")
    model = assemble(scene_xml, spec).compile()
    robot = Robot(model, mujoco.MjData(model), spec)
    robot.home()
    return robot


# --------------------------------------------------------------------------

PRODUCT_PREFIXES = ("near_", "far_", "cage_")


def product_bodies(model) -> dict[str, list[int]]:
    """Product body ids, split into shelf stock and roll-cage stock."""
    shelf, cage = [], []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name.startswith("cage_"):
            cage.append(i)
        elif name.startswith(("near_", "far_")):
            shelf.append(i)
    return {"shelf": shelf, "cage": cage}


def within_reach(robot: Robot) -> tuple[dict, dict]:
    """How much stock each hand could actually get to."""
    groups = product_bodies(robot.model)
    hands = {s: robot.hand(s) for s in robot.spec.sides}
    counts = {}
    for side, hand in hands.items():
        counts[side] = {}
        for label, ids in groups.items():
            if not ids:
                counts[side][label] = 0
                continue
            d = np.linalg.norm(robot.data.xpos[ids] - hand, axis=1)
            counts[side][label] = int((d <= robot.spec.reach).sum())
    return hands, counts


def render_cameras(model, data, out_dir: Path, width=RobotSpec.cam_width,
                   height=RobotSpec.cam_height):
    import imageio.v2 as imageio
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with mujoco.Renderer(model, height=height, width=width) as r:
        for i in range(model.ncam):
            cam = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            r.update_scene(data, camera=cam)
            path = out_dir / f"robot_{cam}.png"
            imageio.imwrite(path, r.render())
            written.append((cam, path))
    return written


def render_rgbd(model, data, camera: str, out_dir: Path,
                width=RobotSpec.cam_width, height=RobotSpec.cam_height,
                far_clip: float = 3.0):
    """Render one camera as RGB and as metric depth.

    MuJoCo returns depth in metres along the camera's optical axis. The raw
    array is saved as .npy because that is what a policy or a point-cloud step
    actually wants; the PNG beside it is only for looking at, normalised over
    `far_clip` so it stays comparable between frames.
    """
    import imageio.v2 as imageio
    out_dir.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(data, camera=camera)
        depth = r.render()
        r.disable_depth_rendering()

    rgb_path = out_dir / f"{camera}_rgb.png"
    npy_path = out_dir / f"{camera}_depth.npy"
    png_path = out_dir / f"{camera}_depth.png"
    imageio.imwrite(rgb_path, rgb)
    np.save(npy_path, depth.astype(np.float32))
    shown = np.clip(depth, 0.0, far_clip) / far_clip
    imageio.imwrite(png_path, (255 * (1.0 - shown)).astype(np.uint8))

    finite = depth[np.isfinite(depth) & (depth < far_clip)]
    stats = dict(min=float(finite.min()) if finite.size else float("nan"),
                 max=float(finite.max()) if finite.size else float("nan"),
                 median=float(np.median(finite)) if finite.size else float("nan"),
                 coverage=float(finite.size) / depth.size)
    return dict(rgb=rgb_path, depth_npy=npy_path, depth_png=png_path, **stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=OUT_DIR / "scene.xml")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--shots", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    robot = build(args.scene)
    m, d = robot.model, robot.data
    print(f"assembled in {time.perf_counter()-t0:.2f} s")
    print(f"  nbody/ngeom/nmesh : {m.nbody} / {m.ngeom} / {m.nmesh}")
    print(f"  njnt / nu / neq   : {m.njnt} / {m.nu} / {m.neq}")

    free = [i for i in range(m.nbody)
            if m.body_jntnum[i] == 1
            and m.jnt_type[m.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE]
    start = d.xpos[free].copy()
    sps = robot.settle(args.seconds)
    drift = np.linalg.norm(d.xpos[free] - start, axis=1) * 1000.0

    print(f"\n--- after {args.seconds:.0f} s settling ---")
    print(f"  contacts        : {d.ncon}")
    print(f"  all finite      : {bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())}")
    print(f"  product drift   : {drift.mean():.2f} mm mean / {drift.max():.2f} mm max")
    print(f"  speed           : {sps:.0f} steps/s = {sps*m.opt.timestep:.2f}x realtime")

    yaw, lift = robot.get_base()
    print(f"\n  base (yaw, lift): {math.degrees(yaw):+.3f} deg, {lift:.4f} m")
    print(f"  arm_mount       : {np.round(robot.mount(), 4)}")
    for side in robot.spec.sides:
        h = robot.hand(side)
        print(f"  {side:5s} hand      : {np.round(h,4)}   "
              f"{np.linalg.norm(h-robot.mount())*1000:.1f} mm from mount")

    hands, counts = within_reach(robot)
    print(f"\n  stock within {robot.spec.reach:.3f} m of each hand:")
    for side in robot.spec.sides:
        c = counts[side]
        print(f"    {side:5s}: {c['shelf']:3d} shelf products, {c['cage']:2d} cage items")

    if args.shots:
        print()
        for cam, path in render_cameras(m, d, OUT_DIR / "shots"):
            print(f"  {cam:20s} -> {path}")

    if args.view:
        # `import mujoco.viewer` would rebind `mujoco` as a local for this
        # whole function, so bind only the submodule
        import mujoco.viewer as mj_viewer
        mj_viewer.launch(m, d)


if __name__ == "__main__":
    main()
