"""Render the figures used by README.md.

Everything in the README is generated from here so the images can be rebuilt
from a clean checkout and stay honest about what the simulator actually looks
like. No hand-compositing, no letterboxing: every panel is rendered at its
camera's native 4:3 and pasted at that aspect.

    python tools/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "figures"

INK = (14, 17, 23)
PANEL = (22, 27, 35)
EDGE = (48, 58, 72)
TEXT = (232, 237, 243)
MUTED = (138, 150, 166)
ACCENT = (242, 143, 58)
COOL = (86, 170, 225)


def font(size: int, bold: bool = False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(model, data, camera, width, height) -> Image.Image:
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera)
        img = r.render()
    # A camera parked inside geometry renders a flat fill and looks like a
    # perfectly good file until you open it. Refuse to save that silently.
    if float(img.std()) < 4.0:
        raise RuntimeError(
            f"camera {camera!r} rendered a near-uniform frame (std "
            f"{img.std():.2f}) -- it is almost certainly inside geometry")
    return Image.fromarray(img)


def free_camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def strip(panels, title, subtitle, pad=26, gap=18, label_h=62) -> Image.Image:
    """Side-by-side panels at their native aspect. No black bars, ever."""
    w, h = panels[0][0].size
    head = 96 if title else 0
    W = pad * 2 + w * len(panels) + gap * (len(panels) - 1)
    H = head + pad + h + label_h + pad
    canvas = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((pad, 26), title, font=font(30, True), fill=TEXT)
        if subtitle:
            d.text((pad, 64), subtitle, font=font(16), fill=MUTED)
    for i, (img, caption, note, accent) in enumerate(panels):
        x = pad + i * (w + gap)
        y = head + pad
        d.rectangle([x - 2, y - 2, x + w + 1, y + h + 1], outline=accent, width=2)
        canvas.paste(img, (x, y))
        d.text((x, y + h + 14), caption, font=font(17, True), fill=accent)
        d.text((x, y + h + 38), note, font=font(14), fill=MUTED)
    return canvas



def product_bbox(rel: str) -> np.ndarray:
    """Full extents of a product, from the reg_bbox marker it ships with."""
    import xml.etree.ElementTree as ET
    xml = REPO / "assets" / "products" / rel / "model.xml"
    for geom in ET.parse(xml).getroot().iter("geom"):
        if geom.get("name") == "reg_bbox":
            return 2.0 * np.array([float(v) for v in geom.get("size").split()])
    raise KeyError(f"{xml}: no reg_bbox")


def grip_quat(rel: str):
    """Rotation that presents the product's narrow side to the jaws.

    The gripper closes along the ee_base y axis. A product whose narrow
    horizontal axis is its local x -- a cereal carton is 34 x 114 mm in plan --
    has to be turned a quarter turn about z, otherwise the jaws close on the
    114 mm face and the box reads as being carried flat rather than pinched on
    its sides the way a hand would hold it.
    """
    bx, by = product_bbox(rel)[:2]
    if bx < by:                      # narrow axis is local x: bring it to y
        return [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], (bx, by, True)
    return [1.0, 0.0, 0.0, 0.0], (bx, by, False)


def jaw_midpoint(model, data, side: str) -> np.ndarray:
    """World point between the two FINGERTIP pads.

    Each jaw carries four collision meshes, collision_00 at the tip through
    collision_03 up at the knuckle. Averaging all four lands ~30 mm up the
    finger, at the palm, which is where a staged object looks jammed into the
    hand instead of pinched between the fingers. Only the tip pads count.
    """
    gn = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    tip = lambda part: [data.geom_xpos[g] for g in range(model.ngeom)
                        if f"{part}_{side}_collision_00" in gn(g)]
    inner, outer = tip("finger_inner"), tip("finger_outer")
    if not inner or not outer:
        raise KeyError(f"no fingertip pads found for {side}")
    return 0.5 * (np.mean(inner, axis=0) + np.mean(outer, axis=0))


def staged(products: dict[str, str], world: str = "bench"):
    """Bench with one product staged in each gripper.

    The product is attached rigidly to the gripper base, not grasped. A real
    closed-loop grasp was tried first and is not what this figure needs: the
    product MJCFs carry RoboCasa's own multi-piece collision meshes and very
    stiff solref, and spawning one between the jaws ejected it (the cereal
    carton left at 14 m). Staging it keeps the figure deterministic and honest,
    and the caption says it is staged.

    Two passes: compile once to find where the jaw midpoint sits in the gripper
    base's own frame, then rebuild with the product parented there.
    """
    import bench as B
    import robot as R
    spec = R.RobotSpec()
    if world == "bench":
        B.build_bench(spec)                   # writes out/bench.xml
        bench_xml = REPO / "out" / "bench.xml"
    else:
        bench_xml = REPO / "out" / "scene.xml"

    probe = R.Robot(*_compiled(R.assemble(bench_xml, spec)), spec)
    probe.home()
    probe.set_base(0.0, 0.45)
    for side in products:
        probe.set_gripper(side, 0.35)
    probe.settle(1.5)
    offsets = {}
    for side in products:
        base = mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_BODY,
                                 f"openarm_{side}_ee_base_link")
        rot = probe.data.xmat[base].reshape(3, 3)
        offsets[side] = rot.T @ (jaw_midpoint(probe.model, probe.data, side)
                                 - probe.data.xpos[base])

    s = R.assemble(bench_xml, spec)
    for side, rel in products.items():
        prod = mujoco.MjSpec.from_file(str(REPO / "assets" / "products" / rel / "model.xml"))
        frame = s.body(f"openarm_{side}_ee_base_link").add_frame()
        frame.pos = offsets[side].tolist()
        quat, (bx, by, turned) = grip_quat(rel)
        frame.quat = quat
        print(f"  {side:5s} {rel:20s} plan {bx*1000:5.1f} x {by*1000:5.1f} mm -> "
              f"jaws take the {min(bx, by)*1000:5.1f} mm side"
              f"{'  (turned 90 deg)' if turned else ''}")
        held = frame.attach_body(prod.body("object"), f"held_{side}_", "")
        for g in held.geoms:
            g.contype, g.conaffinity = 0, 0   # staged, so it must not push the jaws

    bot = R.Robot(*_compiled(s), spec)
    bot.home()
    bot.set_base(0.0, 0.45)
    for side in products:
        bot.set_gripper(side, 0.35)
    bot.settle(1.5)
    for side in products:
        bid = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, f"held_{side}_object")
        gap = np.linalg.norm(bot.data.xpos[bid] - jaw_midpoint(bot.model, bot.data, side))
        print(f"  [{world}] {side:5s} staged {products[side]:20s} "
              f"{gap*1000:5.1f} mm from the jaw midpoint")
    return bot


def _compiled(spec):
    model = spec.compile()
    return model, mujoco.MjData(model)


def sensor_map(body_bot, feed_bot, out_path):
    """The robot front-on, with each camera's feed beside the hardware it is
    bolted to: tower eye centred above the head, one wrist feed per arm.

    A top-down panel used to sit here. It was dropped -- it showed the tower
    footprint and nothing else the front view does not already say.
    """
    robot_w, robot_h = 700, 900
    thumb_w, thumb_h = 480, 360
    pad, gap, label_h, head = 34, 30, 62, 104

    # Body shot comes from the bench: a full front view is impossible inside a
    # 1.30 m aisle, the camera cannot get far enough back. The three feeds come
    # from the aisle, where there is something for them to look at.
    body = render(body_bot.model, body_bot.data,
                  free_camera([0.0, 0.19, 0.95], 2.15, 270, -6), robot_w, robot_h)
    fm, fd = feed_bot.model, feed_bot.data
    eye = render(fm, fd, "tower_eye", thumb_w, thumb_h)
    wl = render(fm, fd, "camera_wrist_left", thumb_w, thumb_h)
    wr = render(fm, fd, "camera_wrist_right", thumb_w, thumb_h)

    W = pad * 2 + thumb_w * 2 + gap * 2 + robot_w
    eye_y = head
    robot_y = eye_y + thumb_h + label_h + gap
    H = robot_y + robot_h + pad

    canvas = Image.new("RGB", (W, H), INK)
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 30), "WHERE EACH CAMERA LIVES", font=font(34, True), fill=TEXT)
    dr.text((pad, 74), "three cameras ride the robot - 640x480 RGB + metric depth - "
                       "body view from the test bench, feeds from the aisle - "
                       "products are staged between the fingertips, not grasped",
            font=font(16), fill=MUTED)

    robot_x = pad + thumb_w + gap
    canvas.paste(body, (robot_x, robot_y))
    anchors = {
        "eye": (robot_x + robot_w * 0.50, robot_y + robot_h * 0.28),
        # a front view mirrors x: the robot's left arm lands on the image right
        "left": (robot_x + robot_w * 0.63, robot_y + robot_h * 0.66),
        "right": (robot_x + robot_w * 0.37, robot_y + robot_h * 0.66),
    }

    def panel(img, x, y, title, note, accent, anchor, edge):
        dr.rectangle([x - 2, y - 2, x + thumb_w + 1, y + thumb_h + 1],
                     outline=accent, width=2)
        canvas.paste(img, (x, y))
        dr.text((x, y + thumb_h + 12), title, font=font(19, True), fill=accent)
        dr.text((x, y + thumb_h + 38), note, font=font(14), fill=MUTED)
        dr.line([edge, anchor], fill=accent, width=2)
        r = 7
        dr.ellipse([anchor[0] - r, anchor[1] - r, anchor[0] + r, anchor[1] + r],
                   outline=accent, width=3)

    # tower eye, centred directly above the head it is mounted on
    eye_x = (W - thumb_w) // 2
    panel(eye, eye_x, eye_y, "TOWER EYE",
          "head camera between the shoulders", ACCENT,
          anchors["eye"], (eye_x + thumb_w // 2, eye_y + thumb_h))

    wrist_y = robot_y + int(robot_h * 0.46)
    panel(wr, pad, wrist_y, "WRIST - ROBOT'S RIGHT",
          "eye-in-hand - cereal carton staged", COOL,
          anchors["right"], (pad + thumb_w, wrist_y + thumb_h // 2))
    right_x = robot_x + robot_w + gap
    panel(wl, right_x, wrist_y, "WRIST - ROBOT'S LEFT",
          "eye-in-hand - jam jar staged", COOL,
          anchors["left"], (right_x, wrist_y + thumb_h // 2))

    canvas.save(out_path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import robot as R
    import bench as B

    bot = R.build()
    m, d = bot.model, bot.data
    # Raise the carriage so the hands sit at the stocked deck. This is a real
    # commanded state (set_base), not a posed one: the arms stay at their
    # qpos=0 home and only the lift moves.
    bot.set_base(0.0, 0.70)
    bot.settle(2.0)

    # ---- hero -------------------------------------------------------------
    # A free camera is the wrong tool here: the aisle is 1.30 m wide, so almost
    # every orbit position lands inside a gondola. `over_shoulder` is already
    # aimed at the restocking gap with the robot in frame, so use it.
    render(m, d, "over_shoulder", 1600, 1000).save(OUT / "hero_aisle.png")

    w, h = 560, 420
    # ---- fixed observation cameras ----------------------------------------
    panels = [
        (render(m, d, "shelf_front", w, h), "SHELF FRONT",
         "the restocking gap and its neighbours", COOL),
        (render(m, d, "overhead", w, h), "OVERHEAD",
         "aisle layout and reachability", COOL),
        (render(m, d, "over_shoulder", w, h), "OVER SHOULDER",
         "robot-relative task context", COOL),
    ]
    strip(panels, "FIXED OBSERVATION CAMERAS",
          "four scene cameras for dataset context and debugging"
          ).save(OUT / "fixed_cameras.png")

    # ---- one annotated figure: robot front-on, feeds beside their hardware --
    kit = {"left": "jam/jam_0", "right": "cereal/cereal_0"}
    sensor_map(staged(kit, "bench"), staged(kit, "aisle"), OUT / "sensor_map.png")

    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(REPO)}  {p.stat().st_size/1000:.0f} kB  "
              f"{Image.open(p).size}")


if __name__ == "__main__":
    main()
