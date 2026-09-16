"""The window: your tracked body on the left, the robot it is driving on the
right, and a readout of everything in between along the bottom.

One window rather than three, because the whole point is watching the two
halves at the same time -- if the robot's elbow is doing something odd you want
your own elbow in the same glance, not on another monitor.

mediapipe 1.0 removed `solutions.drawing_utils`, so the skeleton is drawn here
directly from the landmark arrays. That turns out to be an advantage: the joints
this pipeline actually reads -- shoulders, elbows, wrists, hips -- are drawn
differently from the ones it ignores, so you can see at a glance whether the
tracker has the parts that matter.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .landmarks import P, SIDES, hand_connections, pose_connections

# BGR, because that is what OpenCV wants.
BG = (26, 24, 22)
PANEL = (38, 35, 32)
INK = (232, 232, 235)
DIM = (128, 124, 120)
ACCENT = (70, 160, 250)          # orange-ish: the operator
ROBOT = (150, 220, 120)          # green: the robot's measured state
WARN = (70, 90, 240)             # red
OK = (120, 210, 130)
LEFT_C = (255, 180, 90)          # blue-ish: left side
RIGHT_C = (120, 150, 255)        # red-ish: right side

SIDE_COLOR = {"left": LEFT_C, "right": RIGHT_C}

_POSE_CONN = pose_connections()
_HAND_CONN = hand_connections()

# The landmarks the mapping is actually built from. Everything else is drawn
# faintly, as context.
KEY_POINTS = {
    P.L_SHOULDER: ("L sh", LEFT_C), P.R_SHOULDER: ("R sh", RIGHT_C),
    P.L_ELBOW: ("", LEFT_C), P.R_ELBOW: ("", RIGHT_C),
    P.L_WRIST: ("L hand", LEFT_C), P.R_WRIST: ("R hand", RIGHT_C),
    P.L_HIP: ("", DIM), P.R_HIP: ("", DIM),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.45, color=INK, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _px(pt, w, h):
    return int(round(float(pt[0]) * w)), int(round(float(pt[1]) * h))


# --------------------------------------------------------------------------
# the camera half
# --------------------------------------------------------------------------
def draw_tracking(frame: np.ndarray, obs, min_vis: float = 0.5) -> np.ndarray:
    """Skeleton, hands and the torso frame, drawn over the camera image."""
    img = frame
    h, w = img.shape[:2]
    if obs is None or obs.pose_image is None:
        _text(img, "no pose detected -- step back so your head and shoulders are in frame",
              (16, h - 18), 0.55, WARN)
        return img

    pi = obs.pose_image
    vis = obs.pose_vis if obs.pose_vis is not None else np.ones(len(pi))

    # Bones. A landmark at zero visibility was never found at all, so it has no
    # position to draw -- skip it rather than draw a bone to wherever the
    # placeholder happens to be. Merely *low* visibility is drawn faintly,
    # because seeing the tracker lose your hip is useful.
    for a, b in _POSE_CONN:
        if a >= len(pi) or b >= len(pi):
            continue
        if vis[a] <= 0.01 or vis[b] <= 0.01:
            continue
        seen = vis[a] >= min_vis and vis[b] >= min_vis
        cv2.line(img, _px(pi[a], w, h), _px(pi[b], w, h),
                 INK if seen else (70, 68, 66), 2 if seen else 1, cv2.LINE_AA)

    # every joint, faint
    for i, p in enumerate(pi):
        if vis[i] >= min_vis:
            cv2.circle(img, _px(p, w, h), 2, DIM, -1, cv2.LINE_AA)

    # the ones the retargeter reads, emphasised
    for i, (label, color) in KEY_POINTS.items():
        if vis[i] <= 0.01:
            continue
        pt = _px(pi[i], w, h)
        good = vis[i] >= min_vis
        cv2.circle(img, pt, 7, color if good else WARN, -1, cv2.LINE_AA)
        cv2.circle(img, pt, 7, BG, 1, cv2.LINE_AA)
        if label:
            _text(img, label, (pt[0] + 11, pt[1] + 4), 0.44, color)

    # the shoulder line, which is what the tower's yaw is read off
    if vis[P.L_SHOULDER] >= min_vis and vis[P.R_SHOULDER] >= min_vis:
        a, b = _px(pi[P.L_SHOULDER], w, h), _px(pi[P.R_SHOULDER], w, h)
        cv2.line(img, a, b, ACCENT, 2, cv2.LINE_AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.drawMarker(img, mid, ACCENT, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    # hands
    for side in SIDES:
        hm = obs.hand_image.get(side)
        if hm is None:
            continue
        color = SIDE_COLOR[side]
        for a, b in _HAND_CONN:
            if a < len(hm) and b < len(hm):
                cv2.line(img, _px(hm[a], w, h), _px(hm[b], w, h), color, 1, cv2.LINE_AA)
        for p in hm:
            cv2.circle(img, _px(p, w, h), 2, color, -1, cv2.LINE_AA)
    return img


def draw_calibration_guide(img: np.ndarray, seen: bool) -> None:
    """What to do before anything will move.

    Worth the screen space: with nothing on the picture, an operator has no way
    to know the robot is waiting for them rather than broken.
    """
    h, w = img.shape[:2]
    lines = [
        ("1.  Stand back so your head and shoulders are in frame", INK),
        ("2.  Raise a hand into the CALIBRATE box, hold until it fills", ACCENT),
        ("3.  Drop your arms, face the camera, stand still", ACCENT),
        ("", INK),
        ("Do NOT lean in to press C: leaning turns your shoulder line,", DIM),
        ("and that twist becomes the robot's zero yaw for the session.", DIM),
    ]
    box_h = 26 * len(lines) + 26
    y0 = h - box_h - 12
    box = img[y0:h - 12, 12:w - 12]
    if box.size:
        box[:] = (box * 0.28).astype(np.uint8)
    if not seen:
        _text(img, "looking for you ...", (28, y0 + 24), 0.62, WARN, 2)
    else:
        _text(img, "got you", (28, y0 + 24), 0.62, OK, 2)
    for i, (s, c) in enumerate(lines):
        if s:
            _text(img, s, (28, y0 + 52 + i * 26), 0.52, c)


# --------------------------------------------------------------------------
# air buttons -- pressed with your hand, on camera
# --------------------------------------------------------------------------
class AirButton:
    """A target on the camera image that you press by holding a hand over it.

    This is the only control that does not require you to walk out of shot.
    Reaching for a key -- or a mouse -- means leaning toward the machine, and
    leaning turns one shoulder forward; since the mapping reads torso yaw off
    the shoulder line, calibrating in that pose bakes the twist in as the
    robot's zero. Pressing in mid-air is the fix, because you never stop
    standing where you mean to stand.

    The rectangle is in fractions of the frame, so it lands in the same place
    whatever the camera resolution. `dwell` is how long a hand must stay inside
    before it counts, which is what stops an arm that merely swings past from
    firing it; the ring fills while you hold, so an approach is never silent.
    """

    def __init__(self, label: str, action: str, rect, dwell: float = 1.4,
                 color=ACCENT, compact_rect=None, compact_label=None,
                 compact_dwell: float = 2.4):
        self.label = self.full_label = label
        self.action = action
        self.rect = self.full_rect = rect
        self.dwell = self.full_dwell = dwell
        self.color = color
        # Once you are calibrated you rarely want to do it again, and a big
        # target sitting where you raise your arms is a false trigger waiting
        # to happen. Compact mode shrinks it into the corner and roughly doubles
        # the dwell, so it stays reachable without being in the way.
        self.compact_rect = compact_rect or rect
        self.compact_label = compact_label or label
        self.compact_dwell = compact_dwell
        self.compact = False
        self.hidden = False
        self.progress = 0.0
        self.inside = False
        self._fired = False

    def set_compact(self, on: bool) -> None:
        if on == self.compact:
            return
        self.compact = on
        self.rect = self.compact_rect if on else self.full_rect
        self.label = self.compact_label if on else self.full_label
        self.dwell = self.compact_dwell if on else self.full_dwell
        self.progress = 0.0

    def rect_px(self, w: int, h: int):
        x0, y0, x1, y1 = self.rect
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    def _points(self, obs):
        """Everything that counts as a fingertip, in normalised image coords.

        The 21-point hand mesh is preferred, but it is the first thing to drop
        out when a hand moves fast, so the pose model's own wrist and index
        landmarks stand in. Losing the button mid-press would be worse than
        being a few centimetres less precise about where the press landed.
        """
        pts = []
        if obs is None:
            return pts
        for side in SIDES:
            hm = obs.hand_image.get(side)
            if hm is not None and len(hm) >= 21:
                pts.extend([hm[8][:2], hm[12][:2], hm[0][:2]])   # index, middle, wrist
        if not pts and obs.pose_image is not None:
            vis = obs.pose_vis
            for i in (P.L_INDEX, P.R_INDEX, P.L_WRIST, P.R_WRIST):
                if vis is None or vis[i] >= 0.5:
                    pts.append(obs.pose_image[i][:2])
        return pts

    def update(self, obs, dt: float) -> bool:
        """Advance the dwell timer. Returns True on the frame it fires."""
        if self.hidden:
            self.progress, self.inside = 0.0, False
            return False
        x0, y0, x1, y1 = self.rect
        self.inside = any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1
                          for p in self._points(obs))
        if self.inside:
            self.progress = min(1.0, self.progress + dt / max(self.dwell, 1e-3))
        else:
            # Decay rather than reset, so a momentary tracking dropout does not
            # throw away a press that was nearly complete.
            self.progress = max(0.0, self.progress - dt / max(self.dwell, 1e-3) * 1.5)
            self._fired = False
        if self.progress >= 1.0 and not self._fired:
            self._fired = True
            self.progress = 0.0
            return True
        return False

    def draw(self, img: np.ndarray) -> None:
        if self.hidden:
            return
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.rect_px(w, h)
        panel = img[y0:y1, x0:x1]
        if panel.size:
            panel[:] = (panel * 0.35).astype(np.uint8)
        thick = 3 if self.inside else 2
        cv2.rectangle(img, (x0, y0), (x1, y1), self.color, thick, cv2.LINE_AA)
        # progress along the bottom edge
        if self.progress > 0:
            px = x0 + int((x1 - x0) * self.progress)
            cv2.rectangle(img, (x0, y1 - 8), (px, y1), self.color, -1)
        scale = max(0.5, (x1 - x0) / 260.0)
        (tw, th), _ = cv2.getTextSize(self.label, FONT, scale, 2)
        cv2.putText(img, self.label, (x0 + ((x1 - x0) - tw) // 2,
                                      y0 + ((y1 - y0) + th) // 2 - 8),
                    FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, self.label, (x0 + ((x1 - x0) - tw) // 2,
                                      y0 + ((y1 - y0) + th) // 2 - 8),
                    FONT, scale, INK, 2, cv2.LINE_AA)
        if not self.compact:
            hint = "hold a hand here, then drop it"
            (hw_, _), _ = cv2.getTextSize(hint, FONT, 0.42, 1)
            _text(img, hint, (x0 + ((x1 - x0) - hw_) // 2, y1 - 16), 0.42,
                  self.color if self.inside else DIM)


# --------------------------------------------------------------------------
# widgets
# --------------------------------------------------------------------------
def draw_pose_check(img: np.ndarray, checks: dict) -> None:
    """A live tick-list of what still has to be true before calibrating.

    Shown from across the room, so it is the operator -- not the keyboard --
    that decides when the pose is good.
    """
    rows = [("in frame", checks.get("seen")),
            ("square to camera", checks.get("square")),
            ("shoulders level", checks.get("level")),
            ("arms down", checks.get("arms_down")),
            ("holding still", checks.get("still"))]
    x, y0 = 18, 96
    w = 232
    box = img[y0 - 22:y0 + 26 * len(rows), x - 8:x + w]
    if box.size:
        box[:] = (box * 0.30).astype(np.uint8)
    for i, (label, good) in enumerate(rows):
        y = y0 + i * 26
        cv2.circle(img, (x + 8, y - 4), 6, OK if good else WARN, -1, cv2.LINE_AA)
        _text(img, label, (x + 24, y), 0.52, INK if good else DIM)


def draw_countdown(img: np.ndarray, remaining: float, held: bool = True,
                   grace: bool = False) -> None:
    """Big number over the picture, for calibrating with nobody at the keyboard.

    The whole point of the delay is that you are too far from the machine to
    press anything, so the prompt has to be readable from across the room.
    """
    h, w = img.shape[:2]
    n = max(0, int(math.ceil(remaining)))
    text = str(n) if n > 0 else "GO"
    # During the grace phase the clock runs no matter what, so it must not be
    # coloured as though something were wrong -- you are meant to be moving.
    colour = ACCENT if (held or grace) else WARN
    scale = h / 90.0
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 6)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), FONT, scale,
                (0, 0, 0), 12, cv2.LINE_AA)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), FONT, scale,
                colour, 6, cv2.LINE_AA)
    # The countdown pauses rather than fires on a bad pose, so it has to say
    # which -- a frozen number with no explanation just looks broken.
    if grace:
        msg = "put your arms down"
    elif held:
        msg = "hold it"
    else:
        msg = "waiting: fix the red items on the left"
    (mw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
    cv2.putText(img, msg, ((w - mw) // 2, (h + th) // 2 + 44), FONT, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, msg, ((w - mw) // 2, (h + th) // 2 + 44), FONT, 0.7,
                INK, 1, cv2.LINE_AA)


def bar(img, x, y, w, h, value, lo, hi, label="", cmd=None,
        color=ROBOT, fmt="{:+.2f}", label_w=26):
    """A horizontal gauge, with its label to the left and its value to the right.

    `value` is what the robot measured and fills the bar; `cmd` is what the
    operator asked for and is drawn as a tick. Seeing both is how you tell a
    tracking problem (tick jumping around) from a robot problem (tick steady,
    fill lagging behind it).
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), (58, 54, 50), -1)
    span = max(hi - lo, 1e-9)
    if lo < 0.0 < hi:                      # signed: fill out from zero
        zx = x + int(w * (0.0 - lo) / span)
        vx = x + int(w * (float(value) - lo) / span)
        cv2.rectangle(img, (min(zx, vx), y), (max(zx, vx), y + h), color, -1)
        cv2.line(img, (zx, y), (zx, y + h), (95, 92, 88), 1)
    else:
        vx = x + int(w * (float(value) - lo) / span)
        cv2.rectangle(img, (x, y), (vx, y + h), color, -1)
    if cmd is not None:
        cx = x + int(w * (float(np.clip(cmd, lo, hi)) - lo) / span)
        cv2.line(img, (cx, y - 2), (cx, y + h + 2), ACCENT, 2)
    if label:
        _text(img, label, (x - label_w, y + h - 1), 0.40, DIM)
    _text(img, fmt.format(float(value)), (x + w + 5, y + h - 1), 0.40, INK)


# The on-screen control bar. Each entry is (label, action, key-hint).
#
# These duplicate the keyboard rather than replace it, and both earn their
# place. Once you are standing back far enough to be in frame you can reach
# neither a key nor a mouse, which is what the hands-free countdown is for;
# up close, buttons are discoverable in a way that an unlabelled keymap is not,
# and the keyboard is faster once you know it.
BUTTONS = [
    ("calibrate", "calibrate", "C"),
    ("5s hands-free", "countdown", "T"),
    ("crouch depth", "crouch", "V"),
    ("engage", "engage", "SPACE"),
    ("mirror", "mirror", "M"),
    ("wrist", "wrist", "W"),
    ("reset", "reset", "R"),
    ("follow cam", "follow", "F"),
    ("hide air btn", "hide_air", "H"),
    ("full screen", "fullscreen", "F11"),
    ("quit", "quit", "Q"),
]
BTN_H = 30


def draw_buttons(img, y: int, width: int, active: dict) -> list:
    """Draw the control bar and return [(x0, y0, x1, y1, action), ...] to hit-test.

    Returning the rectangles rather than storing them keeps the drawing and the
    clicking in agreement: there is one set of coordinates, computed once.
    """
    pad, gap = 12, 6
    n = len(BUTTONS)
    bw = max(70, (width - 2 * pad - gap * (n - 1)) // n)
    rects = []
    for i, (label, action, key) in enumerate(BUTTONS):
        x0 = pad + i * (bw + gap)
        x1, y1 = x0 + bw, y + BTN_H
        on = bool(active.get(action))
        cv2.rectangle(img, (x0, y), (x1, y1), (86, 132, 74) if on else (60, 56, 52), -1)
        cv2.rectangle(img, (x0, y), (x1, y1), (96, 92, 88), 1)
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.40, 1)
        _text(img, label, (x0 + max(4, (bw - tw) // 2), y + 14), 0.40,
              INK if on else (198, 196, 200))
        (kw, _), _ = cv2.getTextSize(key, FONT, 0.36, 1)
        _text(img, key, (x0 + max(4, (bw - kw) // 2), y + 26), 0.36, DIM)
        rects.append((x0, y, x1, y1, action))
    return rects


def _pill(img, x, y, text, fg, bg):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.rectangle(img, (x, y - th - 7), (x + tw + 16, y + 7), bg, -1)
    _text(img, text, (x + 8, y), 0.5, fg)
    return x + tw + 16


# --------------------------------------------------------------------------
# the readout
# --------------------------------------------------------------------------
def draw_hud(width: int, height: int, targets, sim_state, info: dict) -> np.ndarray:
    """The strip along the bottom: state, the tower, both arms, the keys."""
    hud = np.full((height, width, 3), PANEL, np.uint8)

    # ---- status line --------------------------------------------------
    y = 26
    if targets is None:
        _pill(hud, 14, y, "STARTING", INK, (70, 66, 62))
    elif targets.driving:
        _pill(hud, 14, y, "DRIVING", (20, 30, 20), OK)
    else:
        _pill(hud, 14, y, targets.reason.upper(), INK, (60, 70, 130))

    bits = [f"cam {info.get('cam_fps', 0):4.1f} fps",
            f"sim {info.get('sim_fps', 0):4.1f} fps",
            f"track {info.get('latency_ms', 0):3.0f} ms",
            # Whether the hips are seen changes how the torso's up axis is
            # found, so it is worth being able to see which mode you are in.
            f"hips {'seen' if info.get('hips') else 'assumed'}",
            f"{info.get('source', 'camera')}"]
    if info.get("mirror"):
        bits.append("MIRRORED")
    if info.get("no_wrist"):
        bits.append("wrist off")
    _text(hud, "   |   ".join(bits), (210, y), 0.46, DIM)

    contacts = sim_state.get("ncon", 0) if sim_state else 0
    _text(hud, f"contacts {contacts}", (width - 130, y), 0.46, DIM)

    # ---- the tower ----------------------------------------------------
    # Fixed-width block on the left; the arms share whatever is left over, so
    # the layout follows the window instead of assuming a width.
    tower_w = 300
    _text(hud, "TOWER", (14, 62), 0.44, ACCENT)
    if targets is not None and sim_state is not None:
        bar(hud, 76, 72, 150, 13, math.degrees(sim_state["yaw"]), -180, 180,
            "yaw", math.degrees(targets.yaw), fmt="{:+6.1f}d")
        bar(hud, 76, 96, 150, 13, sim_state["lift"], 0.0, 0.70,
            "lift", targets.lift, fmt="{:5.3f}m")

    # ---- the arms -----------------------------------------------------
    # Eight gauges per arm -- seven joints and the gripper -- as 4 columns of 2.
    gap = 18
    arm_w = max(220, (width - tower_w - 14 - gap) // 2)
    label_w, val_w, cols = 24, 42, 4
    cell = arm_w // cols
    bw = max(28, cell - label_w - val_w - 6)

    for i, side in enumerate(SIDES):
        x0 = tower_w + i * (arm_w + gap)
        _text(hud, f"{side.upper()} ARM", (x0, 62), 0.44, SIDE_COLOR[side])
        if targets is None or sim_state is None:
            continue
        limits = info.get("limits", {}).get(side)
        q_cmd, q_now = targets.arm[side], sim_state["arm"][side]

        def cell_xy(k):
            return (x0 + (k % cols) * cell + label_w, 72 + (k // cols) * 24)

        for k in range(7):
            bx, by = cell_xy(k)
            lo, hi = (-math.pi, math.pi) if limits is None else limits[k]
            bar(hud, bx, by, bw, 13, math.degrees(q_now[k]),
                math.degrees(lo), math.degrees(hi), f"j{k+1}",
                math.degrees(q_cmd[k]), color=SIDE_COLOR[side], fmt="{:+4.0f}",
                label_w=label_w)
        bx, by = cell_xy(7)
        pin = (targets.raw.get("pinch", {}) or {}).get(side)
        bar(hud, bx, by, bw, 13, targets.grip[side], 0.0, 1.0,
            "grp", color=(90, 200, 230), fmt="{:4.2f}", label_w=label_w)
        _text(hud, "no fingers" if pin is None else f"pinch {pin:4.2f}",
              (x0 + 2 * cell + label_w, by + 12 + 14), 0.40, DIM)

    # ---- the control bar ----------------------------------------------
    rects = draw_buttons(hud, height - BTN_H - 8, width, info.get("active", {}))
    # Camera nudges stay keyboard-only: they are held down and repeated, which
    # a button is bad at, and they sit under the tower block where there is room.
    _text(hud, "robot view:  [ ] turn    , . tilt    - + zoom",
          (14, height - BTN_H - 14), 0.40, DIM)
    return hud, rects


# --------------------------------------------------------------------------
def panel_height_for(screen_w: int, screen_h: int, cam_aspect: float,
                     hud_h: int = 178) -> int:
    """The tallest panel that lets the whole composite fit on the screen.

    The two panels sit side by side, so the layout is limited by width as often
    as by height -- on a 16:9 screen it is always width. Working the size out
    up front and letterboxing the remainder beats letting the window manager
    stretch the image, which would both distort it and put the buttons
    somewhere other than where the mouse says they are.
    """
    by_height = screen_h - hud_h
    by_width = int((screen_w - 2) / (cam_aspect + 1.0))
    return max(200, min(by_height, by_width))


def compose(cam_bgr: np.ndarray, robot_rgb: np.ndarray, targets, sim_state,
            info: dict, panel_h: int = 540, hud_h: int = 178,
            fit: tuple[int, int] | None = None):
    """Camera half, robot half, HUD strip.

    Returns (image, button_rects). The rectangles are in the coordinates of the
    image actually returned -- including any letterbox offset -- so a click at
    (x, y) on screen hit-tests directly against them.
    """
    ch, cw = cam_bgr.shape[:2]
    cam_w = max(1, int(round(cw * panel_h / max(ch, 1))))
    cam = cv2.resize(cam_bgr, (cam_w, panel_h), interpolation=cv2.INTER_AREA)

    rob = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)
    rh, rw = rob.shape[:2]
    rob_w = max(1, int(round(rw * panel_h / max(rh, 1))))
    rob = cv2.resize(rob, (rob_w, panel_h), interpolation=cv2.INTER_AREA)

    top = np.hstack([cam, np.full((panel_h, 2, 3), BG, np.uint8), rob])
    width = top.shape[1]

    _text(top, "OPERATOR", (14, 26), 0.6, ACCENT, 2)
    _text(top, "ROBOT", (cam_w + 18, 26), 0.6, ROBOT, 2)

    hud, rects = draw_hud(width, hud_h, targets, sim_state, info)
    out = np.vstack([top, hud])

    if fit is not None:
        fw, fh = fit
        canvas = np.full((max(fh, out.shape[0]), max(fw, out.shape[1]), 3), BG, np.uint8)
        ox = (canvas.shape[1] - out.shape[1]) // 2
        oy = (canvas.shape[0] - out.shape[0]) // 2
        canvas[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out
        rects = [(x0 + ox, y0 + oy + panel_h, x1 + ox, y1 + oy + panel_h, a)
                 for (x0, y0, x1, y1, a) in rects]
        return canvas, rects

    rects = [(x0, y0 + panel_h, x1, y1 + panel_h, a)
             for (x0, y0, x1, y1, a) in rects]
    return out, rects
