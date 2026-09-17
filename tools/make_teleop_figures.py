"""Render the teleoperation figures used by README.md.

These are deliberately made from the SYNTHETIC operator, not from a person in
front of a camera, and every image says so. The live camera path has run, but it
has not been tested properly yet, so a screenshot of it would present something
unproven as a result. The synthetic operator drives exactly the same retargeting,
filtering and simulation code -- only the landmarks are fabricated -- so these
figures show honestly what the pipeline does with a known input.

Replace them with real teleoperation captures once the live path is tested.

    python tools/make_teleop_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "figures"

from teleop import overlay, synthetic as SY          # noqa: E402
from teleop.config import TeleopConfig               # noqa: E402
from teleop.retarget import Retargeter               # noqa: E402
from teleop.simbridge import SimBridge               # noqa: E402

BG = (23, 17, 14)          # BGR, matching docs/figures
TEXT = (243, 237, 232)
MUTED = (166, 150, 138)
ACCENT = (58, 143, 242)
BANNER = "SYNTHETIC OPERATOR - scripted landmarks, not camera tracking"

# (script time, label). Times are the end of a segment in synthetic.SCRIPT,
# where the fabricated operator has just arrived at that pose.
STAGES = [
    (0.5, "rest: calibrated neutral"),
    (3.0, "left arm forward"),
    (6.0, "right arm out to the side"),
    (12.0, "crouch: carriage drops"),
    (15.0, "turn left: tower yaws"),
    (21.0, "pinch: grippers close"),
]
WINDOW_AT = 9.0             # both arms forward, elbows bent


def _text(img, s, org, scale, color, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def operator_panel(obs, cfg, label: str) -> np.ndarray:
    img = np.full((cfg.cam_height, cfg.cam_width, 3), 30, np.uint8)
    overlay.draw_tracking(img, obs, cfg.min_visibility)
    _text(img, "SYNTHETIC", (14, 30), 0.7, ACCENT, 2)
    _text(img, label, (14, cfg.cam_height - 16), 0.62, TEXT, 1)
    return img


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = TeleopConfig(scene="bench")
    sim = SimBridge(cfg)
    retarget = Retargeter(cfg, sim.limits)
    obs0, _ = SY.scripted(0.0, loop=False)
    obs0.t = 0.0
    retarget.calibrate(obs0)

    dt = 1.0 / 60.0
    wanted = sorted([t for t, _ in STAGES] + [WINDOW_AT])
    grabs: dict[float, tuple] = {}
    t = 0.0
    while t <= max(wanted) + dt:
        obs, _ = SY.scripted(t, loop=False)
        obs.t = t
        targets = retarget(obs)
        sim.apply(targets)
        sim.step(dt)
        for w in wanted:
            if w not in grabs and t >= w:
                grabs[w] = (obs, targets, sim.state(), sim.render())
        t += dt

    # ---- the stages grid ------------------------------------------------
    tiles = []
    for when, label in STAGES:
        obs, targets, state, robot = grabs[when]
        op = cv2.resize(operator_panel(obs, cfg, label), (400, 300),
                        interpolation=cv2.INTER_AREA)
        rob = cv2.resize(cv2.cvtColor(robot, cv2.COLOR_RGB2BGR), (300, 300),
                         interpolation=cv2.INTER_AREA)
        _text(rob, f"yaw {np.degrees(state['yaw']):+.0f} deg", (10, 24), 0.5, TEXT)
        _text(rob, f"lift {state['lift']:.2f} m", (10, 46), 0.5, TEXT)
        tile = np.hstack([op, np.full((300, 4, 3), BG, np.uint8), rob])
        tiles.append(tile)

    gap = 12
    cols = 2
    rows = [np.hstack([tiles[i], np.full((300, gap, 3), BG, np.uint8), tiles[i + 1]])
            for i in range(0, len(tiles), cols)]
    body = rows[0]
    for r in rows[1:]:
        body = np.vstack([body, np.full((gap, body.shape[1], 3), BG, np.uint8), r])
    head = np.full((46, body.shape[1], 3), BG, np.uint8)
    _text(head, BANNER, (12, 31), 0.7, ACCENT, 2)
    grid = np.vstack([head, body, np.full((gap, body.shape[1], 3), BG, np.uint8)])
    grid = cv2.copyMakeBorder(grid, 0, 0, gap, gap, cv2.BORDER_CONSTANT, value=BG)
    path = OUT / "teleop_synthetic_stages.png"
    cv2.imwrite(str(path), grid)
    print(f"wrote {path.relative_to(REPO)}  {grid.shape[1]}x{grid.shape[0]}")

    # ---- one frame of the actual window -----------------------------------
    obs, targets, state, robot = grabs[WINDOW_AT]
    cam = np.full((cfg.cam_height, cfg.cam_width, 3), 30, np.uint8)
    overlay.draw_tracking(cam, obs, cfg.min_visibility)
    _text(cam, "SYNTHETIC OPERATOR", (14, 64), 0.8, ACCENT, 2)
    info = dict(cam_fps=0.0, sim_fps=0.0, latency_ms=0.0, source="synthetic",
                mirror=False, no_wrist=False, limits=sim.limits, hips=True,
                active=dict(engage=True, wrist=True))
    window, _ = overlay.compose(cam, robot, targets, state, info, panel_h=480)
    path = OUT / "teleop_synthetic_window.png"
    cv2.imwrite(str(path), window)
    print(f"wrote {path.relative_to(REPO)}  {window.shape[1]}x{window.shape[0]}")
    sim.close()


if __name__ == "__main__":
    run()
