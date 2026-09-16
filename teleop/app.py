"""Whole-body teleoperation: you drive the lift tower and both arms.

    python -m teleop                       webcam -> robot, one window
    python -m teleop --fullscreen          start filling the screen
    python -m teleop --source synthetic    no camera: a scripted operator
    python -m teleop --viewer              also open the interactive MuJoCo window
    python -m teleop --scene aisle         put the shelves and stock back
    python -m teleop --tracker pose        fastest, but no fingers

--------------------------------------------------------------------------
What drives what
--------------------------------------------------------------------------
    stand up / crouch down    the carriage rides up and down its 0.70 m travel
    turn your shoulders       the tower slews the same way
    move your arms            both 7-DOF arms copy your shoulder, elbow and wrist
    pinch thumb to finger     that side's gripper closes

--------------------------------------------------------------------------
Getting going
--------------------------------------------------------------------------
Get your head and shoulders in frame, face the camera square on with your arms
down, and press C -- or click "5s hands-free" and walk back into shot. That pose
becomes the robot's neutral: the tower's zero yaw, the TOP of the lift, and a
zero wrist. The carriage sits at the top before you engage too, so nothing
lurches the moment it starts driving.

Your hips do not have to be visible -- sitting at a desk is fine, and the HUD
says "hips assumed" when it is standing in an upright torso for you.

Optionally crouch as far as you intend to and press V, which measures your
actual range instead of assuming the default.

SPACE is the deadman -- press it and the arms ease back to rest.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import TeleopConfig
from .landmarks import SIDES
from .quality import PoseCheck, issues


def screen_size(default=(1600, 900)) -> tuple[int, int]:
    """The desktop's size, for laying out full screen."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return default


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m teleop", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TeleopConfig()
    ap.add_argument("--source", choices=("camera", "synthetic"), default="camera",
                    help="'synthetic' runs a scripted operator, no webcam needed")
    ap.add_argument("--tracker", choices=("holistic", "pose"), default=d.tracker,
                    help="holistic (default): body + fingers, 38.6 ms. "
                         "pose: body only, 25.6 ms, jitterier")
    ap.add_argument("--camera", type=int, default=d.camera)
    ap.add_argument("--scene", choices=("bench", "aisle"), default=d.scene,
                    help="bench = the robot on its own (default)")
    ap.add_argument("--fullscreen", action="store_true", help="start full screen")
    ap.add_argument("--viewer", action="store_true",
                    help="also open the interactive MuJoCo window")
    ap.add_argument("--mirror", action="store_true",
                    help="drive the robot as your reflection")
    ap.add_argument("--no-wrist", action="store_true",
                    help="hold the three wrist joints at zero")
    ap.add_argument("--lock-exposure", action="store_true",
                    help="pin a short webcam exposure: about twice the frame "
                         "rate, but fixed at whatever the light is right now")
    ap.add_argument("--pretty", action="store_true", help="shadows and AA (slower)")
    ap.add_argument("--render", type=int, default=d.render_width,
                    help="size of the robot half, in pixels")
    ap.add_argument("--panel", type=int, default=540,
                    help="height of both halves when not full screen")
    ap.add_argument("--headless", type=int, metavar="N", default=0,
                    help="run N control ticks with no window and save the last "
                         "composite frame; a smoke test for the whole pipeline")
    ap.add_argument("--save", default="out/teleop_frame.png",
                    help="where --headless writes its frame")
    ap.add_argument("--record", metavar="PATH", default=None,
                    help="log every tick to PATH.npz and save frames beside it, "
                         "so a session can be reviewed afterwards")
    return ap


def main(argv=None) -> int:
    import cv2

    args = build_parser().parse_args(argv)
    cfg = TeleopConfig(camera=args.camera, tracker=args.tracker, scene=args.scene,
                       mirror=args.mirror, use_wrist=not args.no_wrist,
                       lock_exposure=args.lock_exposure,
                       fullscreen=args.fullscreen,
                       render_width=args.render, render_height=args.render)

    from . import overlay
    from .retarget import Retargeter
    from .simbridge import SimBridge
    from .tracking import make_tracker

    print("building the robot ...")
    sim = SimBridge(cfg, pretty=args.pretty)
    print(f"  {'robot only (tower + arms)' if cfg.scene == 'bench' else 'full aisle'}"
          f" -- nbody {sim.model.nbody}, actuators {sim.model.nu}")

    # The retargeter clamps against the ranges read off the compiled model, so
    # it can never disagree with the robot it is driving.
    retarget = Retargeter(cfg, sim.limits)
    limits = sim.limits

    tracker = make_tracker(cfg, args.source)
    print(f"starting the {args.source} tracker ...")
    try:
        tracker.start()
    except Exception as exc:
        print(f"\ntracker failed to start: {exc}")
        print("try:  python -m teleop --source synthetic   (no camera needed)")
        sim.close()
        return 2

    if args.viewer:
        sim.open_viewer()

    headless = args.headless > 0
    win = "teleop -- operator | robot"
    screen = screen_size()
    fullscreen = False
    clicked: list = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))

    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        print(__doc__.split("Getting going")[1] if "Getting going" in __doc__ else "")

    posecheck = PoseCheck()
    checks: dict = {}
    # One air button, top-right of the operator view. Pressing it starts the
    # countdown rather than calibrating on the spot: you have to raise a hand to
    # reach it, and a raised hand is exactly the pose calibration must not see.
    air = [overlay.AirButton(
        "CALIBRATE", "countdown", (0.63, 0.06, 0.97, 0.30),
        compact_rect=(0.885, 0.04, 0.985, 0.13), compact_label="CAL")]

    rec = {k: [] for k in ("t", "yaw_cmd", "yaw", "lift_cmd", "lift", "driving",
                           "grip_l", "grip_r", "latency", "turn_deg", "motion")}
    rec_arm = {"left": [], "right": []}
    rec_arm_cmd = {"left": [], "right": []}
    rec_frames = []

    targets = None
    last_obs = None
    last_obs_t = None
    last_draw = 0.0
    # Drawing costs ~120 ms a frame full screen; control costs ~2 ms. Tying
    # them together meant sampling a 28 fps tracker 8 times a second and
    # throwing two thirds of it away. They run on separate clocks now.
    draw_interval = 1.0 / 30.0
    rects: list = []
    t_prev = time.perf_counter()
    sim_count, sim_t0, sim_fps = 0, time.perf_counter(), 0.0
    status, status_until = "", 0.0
    countdown_left: float | None = None
    countdown_age = 0.0
    ticks = 0
    running = True

    def flash(msg: str) -> None:
        nonlocal status, status_until
        status, status_until = msg, time.perf_counter() + 2.5
        print(f"  {msg}")

    def set_fullscreen(on: bool) -> None:
        nonlocal fullscreen
        fullscreen = on
        cv2.setWindowProperty(
            win, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    def act(action: str) -> None:
        """One place where every control lands, whether keyed or clicked."""
        nonlocal countdown_left, countdown_age, running
        if action == "calibrate":
            # Refuse a pose that is visibly wrong, and start the hands-free
            # countdown instead -- which is almost certainly what was wanted,
            # since the usual reason the pose is wrong is reaching for this key.
            bad = issues(checks)
            if bad:
                countdown_left, countdown_age = cfg.countdown_s, 0.0
                flash("not a good neutral (" + "; ".join(bad[:2])
                      + ") -- counting down instead")
            elif last_obs is not None and retarget.calibrate(last_obs):
                posecheck.reset()
                flash("calibrated -- this pose is now the robot's neutral")
            else:
                flash("cannot calibrate: get your shoulders in frame")
        elif action == "countdown":
            countdown_left, countdown_age = cfg.countdown_s, 0.0
            flash("calibrating -- put your arms down and stand still")
        elif action == "crouch":
            if last_obs is not None and retarget.set_crouch_bottom(last_obs):
                flash(f"crouch depth set ({retarget.crouch_span:.3f} of frame)")
            else:
                flash("crouch further down, then press V again")
        elif action == "engage":
            retarget.engaged = not retarget.engaged
            flash("ENGAGED" if retarget.engaged else "disengaged -- arms relaxing")
        elif action == "mirror":
            cfg.mirror = not cfg.mirror
            flash(f"mirror {'on' if cfg.mirror else 'off'}")
        elif action == "wrist":
            cfg.use_wrist = not cfg.use_wrist
            flash(f"wrist tracking {'on' if cfg.use_wrist else 'off'}")
        elif action == "reset":
            sim.home()
            retarget.reset_filters(hold=False)   # the robot really did teleport
            flash("robot reset to home, carriage at the top")
        elif action == "hide_air":
            for b in air:
                b.hidden = not b.hidden
            flash("air button " + ("hidden" if air[0].hidden else "shown"))
        elif action == "follow":
            sim.follow = not sim.follow
            flash(f"camera {'follows the carriage' if sim.follow else 'fixed'}")
        elif action == "fullscreen":
            set_fullscreen(not fullscreen)
        elif action == "quit":
            running = False

    if args.fullscreen and not headless:
        set_fullscreen(True)

    try:
        while running:
            now = time.perf_counter()
            dt = min(max(now - t_prev, 1e-4), 0.10)
            t_prev = now

            obs = tracker.latest()
            # Only retarget a frame we have not already seen. Re-running the
            # filters on a duplicate sample feeds them a zero time step and
            # biases them toward whatever the last frame said.
            if obs is not None and obs.t != last_obs_t:
                last_obs, last_obs_t = obs, obs.t
                targets = retarget(obs)
                sim.apply(targets)

            sim.step(dt)
            if not sim.sync():              # the interactive window was closed
                break

            sim_count += 1
            if now - sim_t0 >= 0.5:
                sim_fps = sim_count / (now - sim_t0)
                sim_count, sim_t0 = 0, now

            # ---- hands-free calibration ---------------------------------
            checks = posecheck.update(last_obs, cfg.min_visibility)
            for b in air:
                # Big while it is the thing you need; small once it is not.
                b.set_compact(retarget.cal is not None)
                if b.update(last_obs, dt):
                    act(b.action)
            if countdown_left is not None:
                # Phase one is ungated: you have just had a hand in the air to
                # press the button, and you need time to put it down before
                # anything judges whether your arms are down.
                in_grace = countdown_left > (cfg.countdown_s - cfg.ready_s)
                if in_grace or checks.get("ok"):
                    countdown_left -= dt
                countdown_age += dt
                timed_out = countdown_age >= cfg.calibrate_timeout_s
                if countdown_left <= 0.0 or timed_out:
                    bad = issues(checks)
                    countdown_left = None
                    if last_obs is not None and retarget.calibrate(last_obs):
                        posecheck.reset()
                        if timed_out and bad:
                            flash("calibrated, but " + bad[0]
                                  + " -- press again for a cleaner neutral")
                        else:
                            flash("calibrated -- this pose is now neutral")
                    else:
                        flash("calibration failed -- get your shoulders in frame")

            # ---- draw ----------------------------------------------------
            due = headless or (now - last_draw) >= draw_interval
            if not due:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue
            last_draw = now
            if last_obs is not None and last_obs.frame is not None:
                cam = last_obs.frame.copy()     # copy: the tracker reuses it
            else:
                cam = np.full((cfg.cam_height, cfg.cam_width, 3), 30, np.uint8)
                overlay._text(cam, "waiting for the first tracked frame ...",
                              (24, 48), 0.7, overlay.DIM)
            overlay.draw_tracking(cam, last_obs, cfg.min_visibility)
            for b in air:
                b.draw(cam)
            if retarget.cal is None or countdown_left is not None:
                overlay.draw_pose_check(cam, checks)
            if retarget.cal is None and countdown_left is None:
                overlay.draw_calibration_guide(
                    cam, last_obs is not None
                    and last_obs.visible(cfg.min_visibility))
            if countdown_left is not None:
                overlay.draw_countdown(
                    cam, countdown_left,
                    held=bool(checks.get("ok")),
                    grace=countdown_left > (cfg.countdown_s - cfg.ready_s))
            if status and now < status_until:
                overlay._text(cam, status, (24, 66), 0.62, overlay.ACCENT, 2)

            if fullscreen:
                panel_h = overlay.panel_height_for(
                    screen[0], screen[1], cam.shape[1] / max(cam.shape[0], 1))
                fit = screen
            else:
                panel_h, fit = args.panel, None

            info = dict(cam_fps=tracker.fps, sim_fps=sim_fps,
                        latency_ms=0.0 if last_obs is None else last_obs.latency_ms,
                        source=args.source, mirror=cfg.mirror,
                        no_wrist=not cfg.use_wrist, limits=limits,
                        hips=last_obs is not None
                        and last_obs.hips_visible(cfg.min_visibility),
                        active=dict(engage=retarget.engaged, mirror=cfg.mirror,
                                    wrist=cfg.use_wrist, follow=sim.follow,
                                    fullscreen=fullscreen,
                                    countdown=countdown_left is not None))
            shown, rects = overlay.compose(cam, sim.render(), targets, sim.state(),
                                           info, panel_h=panel_h, fit=fit)

            if args.record and targets is not None:
                st = sim.state()
                rec["t"].append(now)
                rec["yaw_cmd"].append(targets.yaw)
                rec["yaw"].append(st["yaw"])
                rec["lift_cmd"].append(targets.lift)
                rec["lift"].append(st["lift"])
                rec["driving"].append(float(targets.driving))
                rec["grip_l"].append(targets.grip["left"])
                rec["grip_r"].append(targets.grip["right"])
                rec["latency"].append(
                    0.0 if last_obs is None else last_obs.latency_ms)
                rec["turn_deg"].append(float(checks.get("turn_deg", 0.0)))
                rec["motion"].append(min(float(checks.get("motion", 0.0)), 1.0))
                for s_ in SIDES:
                    rec_arm[s_].append(np.asarray(st["arm"][s_], dtype=float))
                    rec_arm_cmd[s_].append(np.asarray(targets.arm[s_], dtype=float))
                if len(rec["t"]) % 45 == 1 and len(rec_frames) < 24:
                    rec_frames.append(shown.copy())

            if headless:
                # Calibrate on the first tracked frame, since there is nobody
                # to press C, then run the requested number of ticks.
                if retarget.cal is None and last_obs is not None:
                    retarget.calibrate(last_obs)
                ticks += 1
                if ticks >= args.headless:
                    path = Path(args.save)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), shown)
                    print(f"\n{ticks} ticks, wrote {path} ({shown.shape[1]}x"
                          f"{shown.shape[0]})")
                    break
                continue

            cv2.imshow(win, shown)
            if not fullscreen:
                # Keep the window the same size as the image, so a click lands
                # where the buttons were drawn.
                cv2.resizeWindow(win, shown.shape[1], shown.shape[0])

            # ---- clicks --------------------------------------------------
            while clicked:
                cx, cy = clicked.pop()
                for x0, y0, x1, y1, action in rects:
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        act(action)
                        break

            # ---- keys ----------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            keymap = {ord("c"): "calibrate", ord("t"): "countdown",
                      ord("v"): "crouch", ord(" "): "engage", ord("m"): "mirror",
                      ord("w"): "wrist", ord("r"): "reset", ord("f"): "follow", ord("h"): "hide_air",
                      ord("q"): "quit", 27: "quit"}
            if key in keymap:
                act(keymap[key])
            elif key == ord("["):
                sim.orbit(d_azimuth=-6.0)
            elif key == ord("]"):
                sim.orbit(d_azimuth=6.0)
            elif key in (ord("-"), ord("_")):
                sim.orbit(d_distance=0.2)
            elif key in (ord("="), ord("+")):
                sim.orbit(d_distance=-0.2)
            elif key == ord(","):
                sim.orbit(d_elevation=-4.0)
            elif key == ord("."):
                sim.orbit(d_elevation=4.0)
            elif key == 0 or key == 255:
                # F11 arrives as a special key on some builds and not at all on
                # others, so the button is the reliable way in.
                pass
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        sim.close()
        cv2.destroyAllWindows()

    if args.record and rec["t"]:
        out = Path(args.record).with_suffix(".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: np.asarray(v) for k, v in rec.items()}
        for s_ in SIDES:
            payload["arm_" + s_] = np.asarray(rec_arm[s_])
            payload["arm_cmd_" + s_] = np.asarray(rec_arm_cmd[s_])
        np.savez_compressed(out, **payload)
        for i, fr in enumerate(rec_frames):
            cv2.imwrite(str(out.with_suffix("")) + "_%02d.png" % i, fr)
        print("recorded %d ticks -> %s (+%d frames)"
              % (len(rec["t"]), out, len(rec_frames)))

    print("\nfinal robot state:")
    st = sim.state()
    print(f"  yaw {np.degrees(st['yaw']):+.1f} deg, lift {st['lift']:.3f} m")
    for s in SIDES:
        print(f"  {s:5s} hand {np.round(st['hand'][s], 3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
