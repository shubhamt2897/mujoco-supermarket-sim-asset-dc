"""Check the teleop maths without a camera, a person, or a window.

    python -m teleop.selftest

Four things get checked, in the order that each depends on the last:

  1. The frame convention. The retargeter assumes the arm's base frame is
     x-forward / y-robot-left / z-up and that the gripper reaches along its own
     -z with the jaws separating along +y. Those were measured off the compiled
     model once; this re-measures them every run, so if the arm model is ever
     swapped the test fails here rather than silently mis-aiming the wrist.

  2. Forward kinematics. `retarget.arm_fk` reimplements the joint chain in
     30 lines of numpy. It is checked against MuJoCo's own FK over random joint
     configurations -- if these agree, the axis signs are right.

  3. The inverse. Random joint angles -> arm_fk -> `solve_arm` -> the same
     angles back. This is the test that catches a flipped sign in the closed
     form, which is otherwise invisible: a wrong sign still produces smooth,
     plausible-looking motion, just mirrored.

  4. The mapping end to end. The fabricated operator in `synthetic.py` is put
     through poses with an obvious right answer -- stand, crouch, turn, raise an
     arm -- and the commands are checked for being the obvious answer.
"""

from __future__ import annotations

import math
import sys

import mujoco
import numpy as np

from . import retarget as RT
from . import synthetic as SY
from .config import TeleopConfig
from .landmarks import SIDES
from .simbridge import SimBridge


class Report:
    def __init__(self):
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        mark = "pass" if ok else "FAIL"
        line = f"  [{mark}] {label}"
        if detail:
            line += f"   {detail}"
        print(line)
        if not ok:
            self.failures.append(label)
        return ok

    def close(self, obs=None, tol=None) -> bool:
        pass


def _mj_arm_frames(sim: SimBridge, side: str, q) -> dict:
    """MuJoCo's own answer for the same joint angles, in the arm base frame."""
    m, d = sim.model, sim.data
    d.qpos[sim.bot.arm_qadr[side]] = np.asarray(q, dtype=float)
    mujoco.mj_forward(m, d)

    def bid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

    base = bid(f"openarm_{side}_base_link")
    r_base = d.xmat[base].reshape(3, 3)

    def in_base_dir(a, b):
        v = d.xpos[bid(b)] - d.xpos[bid(a)]
        return r_base.T @ (v / max(np.linalg.norm(v), 1e-12))

    ee = bid(f"openarm_{side}_ee_base_link")
    r07 = r_base.T @ d.xmat[ee].reshape(3, 3)
    inner = d.xpos[bid(f"openarm_{side}_ee_inner_finger")]
    outer = d.xpos[bid(f"openarm_{side}_ee_outer_finger")]
    jaw = r_base.T @ ((outer - inner) / max(np.linalg.norm(outer - inner), 1e-12))
    return dict(
        # link2 -> link4 runs down the upper arm; link4 -> link6 down the forearm
        upper=in_base_dir(f"openarm_{side}_link2", f"openarm_{side}_link4"),
        fore=in_base_dir(f"openarm_{side}_link4", f"openarm_{side}_link6"),
        r07=r07, jaw=jaw, r_base=r_base,
    )


def test_frames(sim: SimBridge, rep: Report) -> None:
    print("\n1. frame conventions, measured off the compiled model")
    m, d = sim.model, sim.data
    sim.home()
    mujoco.mj_forward(m, d)
    for side in SIDES:
        f = _mj_arm_frames(sim, side, np.zeros(7))
        rb = f["r_base"]
        # Columns of r_base are the arm base axes expressed in world. At yaw 0
        # the robot faces world +y.
        rep.check(np.allclose(rb[:, 0], [0, 1, 0], atol=1e-6),
                  f"{side}: base x is forward (world +y)", str(np.round(rb[:, 0], 4)))
        want_left = [-1, 0, 0] if side == "left" else [-1, 0, 0]
        rep.check(np.allclose(rb[:, 1], want_left, atol=1e-6),
                  f"{side}: base y is the robot's left (world -x)",
                  str(np.round(rb[:, 1], 4)))
        rep.check(np.allclose(rb[:, 2], [0, 0, 1], atol=1e-6),
                  f"{side}: base z is up", str(np.round(rb[:, 2], 4)))
        rep.check(np.allclose(f["upper"], [0, 0, -1], atol=1e-6),
                  f"{side}: upper arm hangs down at q=0", str(np.round(f["upper"], 4)))
        # The gripper reaches along its own -z ...
        approach = f["r07"] @ np.array([0.0, 0.0, -1.0])
        rep.check(np.allclose(approach, [0, 0, -1], atol=1e-6),
                  f"{side}: gripper approaches along its -z",
                  str(np.round(approach, 4)))
        # ... and the jaws separate along its +y (sign is per side, axis is not)
        rep.check(abs(abs(float(f["jaw"] @ (f["r07"] @ np.array([0.0, 1.0, 0.0])))) - 1.0) < 1e-6,
                  f"{side}: jaws separate along gripper y", str(np.round(f["jaw"], 4)))
    sim.home()


def _random_q(side: str, rng, limits, principal: bool = False) -> np.ndarray:
    """A random joint configuration inside the limits.

    `principal` additionally keeps shoulder abduction inside +/-90 deg. The arm
    can swing past that, but every such pose has a second set of angles that
    aims it identically, and the solver deliberately always returns the +/-90
    one (see solve_shoulder_elbow). Angles only round-trip as *numbers* on this
    branch; off it, only the resulting pose can be compared.
    """
    lo, hi = limits[side][:, 0], limits[side][:, 1].copy()
    lo = lo.copy()
    # Stay a little inside the limits: right at them the arm self-collides, and
    # MuJoCo's FK is still exact but the pose is not one teleop would produce.
    q = lo + (hi - lo) * rng.uniform(0.08, 0.92, size=7)
    if principal:
        q[1] = rng.uniform(max(lo[1], -math.pi / 2 * 0.97),
                           min(hi[1], math.pi / 2 * 0.97))
    return q


def test_fk(sim: SimBridge, rep: Report, n: int = 200) -> None:
    print(f"\n2. forward kinematics vs MuJoCo, {n} random configurations per arm")
    rng = np.random.default_rng(0)
    limits = sim.limits
    for side in SIDES:
        worst = dict(upper=0.0, fore=0.0, r07=0.0)
        for _ in range(n):
            q = _random_q(side, rng, limits)
            mine = RT.arm_fk(q, side)
            theirs = _mj_arm_frames(sim, side, q)
            worst["upper"] = max(worst["upper"],
                                 float(np.abs(mine["upper"] - theirs["upper"]).max()))
            worst["fore"] = max(worst["fore"],
                                float(np.abs(mine["fore"] - theirs["fore"]).max()))
            worst["r07"] = max(worst["r07"],
                               float(np.abs(mine["r07"] - theirs["r07"]).max()))
        rep.check(worst["upper"] < 1e-9, f"{side}: upper arm direction",
                  f"max err {worst['upper']:.2e}")
        rep.check(worst["fore"] < 1e-9, f"{side}: forearm direction",
                  f"max err {worst['fore']:.2e}")
        rep.check(worst["r07"] < 1e-9, f"{side}: gripper rotation",
                  f"max err {worst['r07']:.2e}")
    sim.home()


def test_ik(sim: SimBridge, rep: Report, n: int = 2000) -> None:
    print(f"\n3. the closed-form inverse round-trips, {n} configurations per arm")
    rng = np.random.default_rng(1)
    limits = sim.limits
    def pose_err(a: dict, b: dict) -> float:
        return max(float(np.abs(a["upper"] - b["upper"]).max()),
                   float(np.abs(a["fore"] - b["fore"]).max()),
                   float(np.abs(a["r07"] - b["r07"]).max()))

    for side in SIDES:
        # (a) On the branch the solver targets, the numbers themselves come back.
        worst = 0.0
        for _ in range(n):
            q = _random_q(side, rng, limits, principal=True)
            fk = RT.arm_fk(q, side)
            back = RT.solve_arm(fk["upper"], fk["fore"], fk["r07"], side, prev=q)
            # Modulo a full turn: -190 deg and +170 deg name the same joint
            # position, and _wrap_into may legitimately return either.
            worst = max(worst, float(np.abs(np.angle(np.exp(1j * (back - q)))).max()))
        rep.check(worst < 1e-8, f"{side}: joint angles recovered exactly",
                  f"max err {math.degrees(worst):.2e} deg")

        # (b) Anywhere in the full joint range -- including the poses whose
        # angles come back as the other branch -- the arm must still end up
        # pointing the same way.
        worst = 0.0
        for _ in range(n):
            q = _random_q(side, rng, limits)
            fk = RT.arm_fk(q, side)
            back = RT.solve_arm(fk["upper"], fk["fore"], fk["r07"], side, prev=q)
            worst = max(worst, pose_err(RT.arm_fk(back, side), fk))
        rep.check(worst < 1e-8, f"{side}: arm pose reproduced over the whole range",
                  f"max err {worst:.2e}")

        # (c) And again through the two directions the tracker actually
        # supplies, rather than a rotation matrix it never has.
        worst = 0.0
        for _ in range(n // 2):
            q = _random_q(side, rng, limits)
            fk = RT.arm_fk(q, side)
            r_grip = RT.gripper_frame(fk["approach"], fk["jaw"])
            back = RT.solve_arm(fk["upper"], fk["fore"], r_grip, side, prev=q)
            worst = max(worst, pose_err(RT.arm_fk(back, side), fk))
        rep.check(worst < 1e-8, f"{side}: rebuilt from approach+jaw directions",
                  f"max err {worst:.2e}")
    sim.home()


def test_limits(rep: Report) -> None:
    print("\n3b. joint limits and the +/-360 deg aliasing")
    lim = RT.DEFAULT_LIMITS["left"]
    # +170 deg on joint1 is the same pose as -190 deg, which IS inside the
    # left arm's -200..+80 range. Clamping to +80 would lose an overhead reach.
    got = RT._wrap_into(math.radians(170.0), lim[0, 0], lim[0, 1])
    rep.check(abs(got - math.radians(-190.0)) < 1e-9,
              "joint1 +170 deg aliases to the reachable -190 deg",
              f"{math.degrees(got):.1f} deg")
    got = RT._wrap_into(math.radians(120.0), lim[0, 0], lim[0, 1])
    rep.check(abs(got - lim[0, 1]) < 1e-9,
              "joint1 +120 deg is genuinely out of range, so it clamps",
              f"{math.degrees(got):.1f} deg")
    q = RT.clamp_to_limits(np.full(7, 9.0), lim)
    rep.check(bool(np.all(q >= lim[:, 0] - 1e-9) and np.all(q <= lim[:, 1] + 1e-9)),
              "everything lands inside the limits")


def test_mapping(sim: SimBridge, rep: Report) -> None:
    print("\n4. the operator mapping, on the fabricated operator")
    cfg = TeleopConfig()
    r = RT.Retargeter(cfg, sim.limits)

    rest = SY.make_observation(0.0)
    rep.check(r.calibrate(rest), "calibrates from the rest pose")
    rep.check(r.engaged, "engages on calibration")

    def run(obs, n=140):
        """Settle the filters, which are rate limited and would otherwise still
        be on their way to the answer."""
        out = None
        for i in range(n):
            o = SY.Observation(t=obs.t + i * 0.02, pose_world=obs.pose_world,
                               pose_image=obs.pose_image, pose_vis=obs.pose_vis,
                               hand_world=obs.hand_world, hand_image=obs.hand_image)
            out = r(o)
        return out

    # --- standing: carriage at the top ---
    t = run(SY.make_observation(1.0))
    rep.check(abs(t.lift - cfg.lift_top) < 1e-3,
              "standing puts the carriage at the top", f"lift {t.lift:.3f} m")
    rep.check(abs(t.yaw) < 1e-3, "square to the camera means no yaw",
              f"yaw {math.degrees(t.yaw):+.2f} deg")
    rep.check(float(np.abs(t.arm['left']).max()) < math.radians(2.0),
              "arms down maps to the robot's rest pose",
              f"max |q| {math.degrees(float(np.abs(t.arm['left']).max())):.2f} deg")

    # --- crouching: carriage drops ---
    t = run(SY.make_observation(2.0, crouch=0.44))
    rep.check(t.lift < cfg.lift_bottom + 0.02,
              "a full crouch puts the carriage at the bottom", f"lift {t.lift:.3f} m")
    t_half = run(SY.make_observation(3.0, crouch=0.22))
    rep.check(abs(t_half.lift - 0.35) < 0.05,
              "half a crouch is half the travel", f"lift {t_half.lift:.3f} m")

    # --- turning: the tower slews the same way ---
    t = run(SY.make_observation(4.0, yaw=math.radians(30.0)))
    rep.check(t.yaw > math.radians(20.0),
              "turning to your left yaws the tower to its left",
              f"yaw {math.degrees(t.yaw):+.1f} deg for a 30 deg turn")
    t = run(SY.make_observation(5.0, yaw=math.radians(-30.0)))
    rep.check(t.yaw < math.radians(-20.0), "and to the right, the other way",
              f"yaw {math.degrees(t.yaw):+.1f} deg")

    # --- raising an arm: the right arm, by the right amount, alone ---
    for side in SIDES:
        other = "right" if side == "left" else "left"
        obs = SY.make_observation(6.0, upper={side: SY.arm_dir(90, 0, side)},
                                  fore={side: SY.arm_dir(90, 0, side)})
        t = run(obs)
        q1 = t.arm[side][0]
        expect = -math.pi / 2 if side == "left" else math.pi / 2
        rep.check(abs(q1 - expect) < math.radians(6.0),
                  f"{side} arm forward drives {side} joint1 to {math.degrees(expect):+.0f} deg",
                  f"got {math.degrees(q1):+.1f} deg")
        rep.check(float(np.abs(t.arm[other][:2]).max()) < math.radians(6.0),
                  f"and leaves the {other} arm alone",
                  f"max |q| {math.degrees(float(np.abs(t.arm[other][:2]).max())):.1f} deg")

        obs = SY.make_observation(7.0, upper={side: SY.arm_dir(0, 90, side)},
                                  fore={side: SY.arm_dir(0, 90, side)})
        t = run(obs)
        q2 = t.arm[side][1]
        expect = -math.pi / 2 if side == "left" else math.pi / 2
        rep.check(abs(q2 - expect) < math.radians(8.0),
                  f"{side} arm out to the side drives joint2",
                  f"got {math.degrees(q2):+.1f} deg")

    # --- elbow ---
    obs = SY.make_observation(8.0,
                              upper={"left": SY.arm_dir(90, 0, "left")},
                              fore={"left": SY.arm_dir(0, 0, "left")})
    t = run(obs)
    rep.check(abs(t.arm["left"][3] - math.pi / 2) < math.radians(8.0),
              "arm forward with the forearm hanging = 90 deg of elbow",
              f"got {math.degrees(t.arm['left'][3]):+.1f} deg")

    # --- gripper ---
    t = run(SY.make_observation(9.0, pinch={"left": 0.20, "right": 0.20}))
    rep.check(t.grip["left"] < 0.05, "a pinch closes the jaws",
              f"grip {t.grip['left']:.3f}")
    t = run(SY.make_observation(10.0, pinch={"left": 1.40, "right": 1.40}))
    rep.check(t.grip["left"] > 0.95, "an open hand opens them",
              f"grip {t.grip['left']:.3f}")

    # --- seated at a desk: hips cropped or hidden, shoulders fine ---
    # This is how the teleop is usually tested, and requiring hips meant it
    # simply never engaged. The shoulders alone must be enough.
    seated = SY.make_observation(12.0, upper={"left": SY.arm_dir(90, 0, "left")},
                                 fore={"left": SY.arm_dir(90, 0, "left")})
    seated.pose_vis = seated.pose_vis.copy()
    seated.pose_vis[[23, 24]] = 0.25            # hips, as MediaPipe reports them
    rep.check(seated.visible(cfg.min_visibility), "a seated operator still counts as visible")
    rep.check(not seated.hips_visible(cfg.min_visibility), "but the hips do not")
    r2 = RT.Retargeter(cfg, sim.limits)
    rest_seated = SY.make_observation(0.0)
    rest_seated.pose_vis = rest_seated.pose_vis.copy()
    rest_seated.pose_vis[[23, 24]] = 0.25
    rep.check(r2.calibrate(rest_seated), "calibrates with the hips out of frame")
    out = None
    for i in range(140):
        o = SY.Observation(t=12.0 + i * 0.02, pose_world=seated.pose_world,
                           pose_image=seated.pose_image, pose_vis=seated.pose_vis,
                           hand_world=seated.hand_world, hand_image=seated.hand_image)
        out = r2(o)
    rep.check(out.driving, "and drives from a chair", out.reason)
    rep.check(abs(out.arm["left"][0] + math.pi / 2) < math.radians(6.0),
              "with the arm still going where it should",
              f"joint1 {math.degrees(out.arm['left'][0]):+.1f} deg")

    # --- recalibrating must not step the commands ---
    # Found in a recorded session: recalibrating at the bottom of a crouch
    # rebuilt the filters, the fresh rate limiter had nothing to limit against,
    # and the lift command jumped 0.000 -> 0.700 m in one tick (9.998 m/s
    # against a 0.6 m/s limit). On real hardware that is a full-travel lunge.
    r3 = RT.Retargeter(cfg, sim.limits)
    crouched = SY.make_observation(0.0, crouch=0.44)
    r3.calibrate(SY.make_observation(0.0))
    out = None
    for i in range(160):
        o = SY.Observation(t=i * 0.02, pose_world=crouched.pose_world,
                           pose_image=crouched.pose_image,
                           pose_vis=crouched.pose_vis,
                           hand_world=crouched.hand_world,
                           hand_image=crouched.hand_image)
        out = r3(o)
    low = out.lift
    rep.check(low < 0.05, "crouched right down before recalibrating",
              f"lift {low:.3f} m")
    # Now recalibrate while still crouched: standing height is redefined, so
    # the raw target leaps to the top of the travel.
    r3.calibrate(crouched)
    t0 = 160 * 0.02
    worst_rate = 0.0
    prev = low
    for i in range(12):
        o = SY.Observation(t=t0 + i * 0.02, pose_world=crouched.pose_world,
                           pose_image=crouched.pose_image,
                           pose_vis=crouched.pose_vis,
                           hand_world=crouched.hand_world,
                           hand_image=crouched.hand_image)
        out = r3(o)
        worst_rate = max(worst_rate, abs(out.lift - prev) / 0.02)
        prev = out.lift
    rep.check(worst_rate <= cfg.lift_rate * 1.15,
              "and the lift still slews, rather than stepping",
              f"peak {worst_rate:.3f} m/s against a {cfg.lift_rate:.2f} limit")

    # --- the deadman ---
    r.engaged = False
    t = run(SY.make_observation(11.0, upper={"left": SY.arm_dir(90, 0, "left")}))
    rep.check(not t.driving, "disengaged stops driving", t.reason)
    rep.check(float(np.abs(t.arm["left"]).max()) < 1e-6,
              "and the arms ease back to rest")


def test_sim(sim: SimBridge, rep: Report) -> None:
    print("\n5. the commands actually move the robot")
    cfg = TeleopConfig()
    r = RT.Retargeter(cfg, sim.limits)
    r.calibrate(SY.make_observation(0.0))
    sim.home()

    start = {s: sim.bot.hand(s).copy() for s in SIDES}
    t_end = None
    for i in range(400):                     # ~6.7 s at 60 Hz
        obs = SY.make_observation(
            i / 60.0, crouch=0.44,
            upper={s: SY.arm_dir(85, 12, s) for s in SIDES},
            fore={s: SY.arm_dir(85, 12, s) for s in SIDES})
        t_end = r(obs)
        sim.apply(t_end)
        sim.step(1.0 / 60.0)

    yaw, lift = sim.bot.get_base()
    rep.check(abs(lift - t_end.lift) < 0.02,
              "the carriage reaches the commanded lift",
              f"cmd {t_end.lift:.3f} m, got {lift:.3f} m")
    for s in SIDES:
        moved = float(np.linalg.norm(sim.bot.hand(s) - start[s]))
        rep.check(moved > 0.15, f"the {s} hand moved with the operator",
                  f"{moved*1000:.0f} mm")
        err = float(np.abs(sim.bot.get_arm(s) - t_end.arm[s]).max())
        rep.check(err < math.radians(6.0), f"the {s} arm tracks its command",
                  f"max joint err {math.degrees(err):.2f} deg")
    rep.check(bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()),
              "the simulation stayed finite")
    sim.home()


def main() -> int:
    rep = Report()
    print("building the robot (tower + arms, nothing else) ...")
    sim = SimBridge(TeleopConfig(scene="bench"))
    print(f"  nbody {sim.model.nbody}, nu {sim.model.nu}, "
          f"timestep {sim.model.opt.timestep*1000:.1f} ms")

    test_frames(sim, rep)
    test_fk(sim, rep)
    test_ik(sim, rep)
    test_limits(rep)
    test_mapping(sim, rep)
    test_sim(sim, rep)
    sim.close()

    print(f"\n{'-'*66}")
    if rep.failures:
        print(f"{len(rep.failures)} of {rep.checks} checks FAILED:")
        for f in rep.failures:
            print(f"  - {f}")
        return 1
    print(f"all {rep.checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
