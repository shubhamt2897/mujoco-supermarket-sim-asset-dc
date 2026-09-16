"""Generate the supermarket shelf-restocking scene as MJCF.

Two gondola runs face each other across the aisle, the way they do in a store.
Frame convention, in metres:
    +x  runs along the gondola
    +y  points into the near run; its bay face is the plane y = 0, so the aisle
        is negative y and the far run faces back across it
    +z  is up

Run `python scene.py` to write out/scene.xml, then `--view` to open it.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from product_library import Product, load_library          # noqa: E402
from product_spec import PRODUCTS                          # noqa: E402

REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "out"

# Products are seated this far above the surface they rest on. Enough to keep
# the solver out of initial penetration, small enough that the resulting settle
# does not read as drift.
SEAT_CLEARANCE = 0.0002


# --------------------------------------------------------------------------
# dimensions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ShelfDims:
    """A gondola run: `n_bays` standard bays bolted end to end.

    Every figure here is a real supermarket dimension. A single 1.25 m bay on
    its own reads as a kitchen unit; a run is three or four of them, which is
    why n_bays exists rather than a longer bay.
    """
    bay_length: float = 1.25
    n_bays: int = 3
    shelf_depth: float = 0.47
    rack_height: float = 1.80
    deck_heights: tuple[float, ...] = (0.15, 0.60, 1.05, 1.50)
    stocked_decks: tuple[int, ...] = (2, 3)      # indices into deck_heights
    board_thickness: float = 0.018
    front_lip: float = 0.025
    front_lip_thickness: float = 0.012
    kick_plate: float = 0.12
    aisle_width: float = 1.30

    upright_thickness: float = 0.040
    back_panel_thickness: float = 0.018
    price_rail_height: float = 0.030
    price_rail_thickness: float = 0.006

    @property
    def run_length(self) -> float:
        return self.bay_length * self.n_bays

    @property
    def half_run(self) -> float:
        return self.run_length / 2.0

    def bay_span(self, bay: int) -> tuple[float, float]:
        """Stockable x range inside one bay, between its two uprights."""
        x0 = -self.half_run + bay * self.bay_length
        return (x0 + self.upright_thickness, x0 + self.bay_length - self.upright_thickness)

    def deck_clearance(self, deck_index: int) -> float:
        """Vertical room above a deck, up to the next deck or the rack top."""
        z = self.deck_heights[deck_index]
        if deck_index + 1 < len(self.deck_heights):
            return self.deck_heights[deck_index + 1] - self.board_thickness - z
        return self.rack_height - z


@dataclass(frozen=True)
class StockingDims:
    facing_gap: float = 0.006        # side-to-side slack between facings
    row_gap: float = 0.008           # front-to-back slack between rows
    front_setback: float = 0.008     # clear air behind the front lip
    back_setback: float = 0.010      # gap in front of the back panel
    max_rows_deep: int = 3           # front row is dynamic, the rest are static
    min_facings_per_block: int = 3   # a SKU gets a contiguous run of facings
    max_facings_per_block: int = 5
    height_margin: float = 0.020     # refuse product taller than this under the deck


@dataclass(frozen=True)
class GapSpec:
    """The hole in the shelf the robot is meant to fill.

    Given in world x so it can be parked squarely in front of the robot, which
    is the only place a restocking gap is any use.
    """
    deck_index: int = 2
    center_x: float = 0.0
    width: float = 0.42

    @property
    def interval(self) -> tuple[float, float]:
        return (self.center_x - self.width / 2.0, self.center_x + self.width / 2.0)


@dataclass(frozen=True)
class ShelfRun:
    """One gondola run, placed and turned in the world.

    yaw 0 puts the bay face on y = 0 looking down -y; yaw pi turns the run
    around so it faces back across the aisle.
    """
    name: str
    origin: tuple[float, float]
    yaw: float
    categories: tuple[tuple[str, ...], ...]
    stocked: bool = True
    dynamic_front_row: bool = True
    rows_deep: int = 3
    gap: GapSpec | None = None
    # Scenery across the aisle needs no collision geometry at all: it is out of
    # the arm's reach, so nothing can ever touch it.
    collides: bool = True

    def to_world(self, p_local) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y, z = (float(v) for v in p_local)
        return np.array([self.origin[0] + c * x - s * y,
                         self.origin[1] + s * x + c * y,
                         z])


@dataclass(frozen=True)
class RollCageDims:
    """The wheeled stock cage shops restock from.

    Not a customer trolley: the point is the top tray at 1.10 m, which is
    inside the arm's vertical envelope, whereas a trolley basket is not.
    """
    length: float = 0.80             # along x
    width: float = 0.70              # along y
    top_tray_height: float = 1.10
    base_tray_height: float = 0.18
    tray_thickness: float = 0.020
    upright_section: float = 0.030
    panel_bar_thickness: float = 0.012
    n_panel_bars: int = 3
    caster_radius: float = 0.050
    caster_width: float = 0.028
    n_loose_products: int = 8
    # Clear of the tower's slew circle. The arms sweep a ~0.35 m radius about
    # the tower axis; with the cage at x = 0.66 its near edge sat 0.26 m out and
    # the right arm fouled it at -14 deg of yaw, so the base could not turn.
    # At x = 0.86 the near edge is 0.46 m out and yaw is free through +/-180 deg.
    # The cage no longer needs to be in reach at yaw = 0 -- that is what the
    # rotary joint is for: the robot turns to face the cage, then unloads it.
    pos_x: float = 0.86
    pos_y: float = -0.56            # negative = out in the aisle


@dataclass(frozen=True)
class RobotMountDims:
    """Where the lift tower stands. Its geometry comes from tower.xml."""
    standoff: float = 0.45          # out from the near bay face, into the aisle
    pos_x: float = 0.0
    # Measured reach of the OpenArm v2 from the shoulder (see robot.py), plus
    # margin. Front-row stock outside this band can never be touched, so it is
    # left static: a free body the arm cannot reach costs solver time and buys
    # nothing.
    reach_radius: float = 0.589
    dynamic_margin: float = 0.18


@dataclass(frozen=True)
class FloorDims:
    """Open shop floor. No room, no walls -- a gondola aisle is not a box."""
    half_extent: float = 9.00
    aisle_light_height: float = 2.70
    n_aisle_lights: int = 4


@dataclass(frozen=True)
class SceneConfig:
    shelf: ShelfDims = field(default_factory=ShelfDims)
    stocking: StockingDims = field(default_factory=StockingDims)
    cage: RollCageDims = field(default_factory=RollCageDims)
    robot: RobotMountDims = field(default_factory=RobotMountDims)
    floor: FloorDims = field(default_factory=FloorDims)
    seed: int = 7

    @property
    def runs(self) -> tuple[ShelfRun, ...]:
        # heavier, rounder stock low; cartons up top, the way a real bay is blocked
        near = (("canned_food", "can", "jam", "yogurt"),
                ("cereal", "boxed_food", "milk", "boxed_drink"))
        far = (("cereal", "boxed_food", "jam", "can"),
               ("boxed_drink", "milk", "canned_food", "yogurt"))
        return (
            ShelfRun("near", (0.0, 0.0), 0.0, near,
                     dynamic_front_row=True, rows_deep=3, gap=GapSpec()),
            # the far run is scenery: all static, shallower, so it costs almost
            # nothing to simulate while still reading as a stocked aisle
            ShelfRun("far", (0.0, -self.shelf.aisle_width), math.pi, far,
                     dynamic_front_row=False, rows_deep=2, gap=None,
                     collides=False),
        )


# --------------------------------------------------------------------------
# XML helpers
# --------------------------------------------------------------------------

def fmt(*vals) -> str:
    out = []
    for v in vals:
        for x in (v if isinstance(v, (list, tuple, np.ndarray)) else [v]):
            out.append(f"{float(x):.6g}")
    return " ".join(out)


def E(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        if v is None:
            continue
        el.set(k.rstrip("_"), v if isinstance(v, str) else fmt(v))
    return el


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF `xyaxes` aiming a camera from eye at target.

    A MuJoCo camera looks down its own -z with +x right and +y up, so xyaxes is
    just (right, true_up). Computing it beats hand-writing axis triples.
    """
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError("look_at: view direction is parallel to `up`")
    right /= norm
    return fmt(right, np.cross(right, fwd))


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def box_inertia(mass: float, extents: np.ndarray) -> np.ndarray:
    """Diagonal inertia of a solid box, good enough for a packaged good."""
    x, y, z = extents
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


# --------------------------------------------------------------------------
# stocking plan
# --------------------------------------------------------------------------

@dataclass
class Placement:
    product: Product
    pos: np.ndarray          # world
    yaw: float               # world
    dynamic: bool
    collide: bool
    run: str
    deck_index: int
    name: str


def plan_bay_deck(cfg: SceneConfig, run: ShelfRun, bay: int, deck_index: int,
                  categories, library, rng, counter: list[int]):
    """Face up one deck of one bay, left to right, skipping the gap."""
    s, st = cfg.shelf, cfg.stocking
    x_lo, x_hi = s.bay_span(bay)
    y_front = s.front_lip_thickness + st.front_setback
    y_back = s.shelf_depth - s.back_panel_thickness - st.back_setback
    deck_z = s.deck_heights[deck_index]
    clearance = s.deck_clearance(deck_index) - st.height_margin

    gap = None
    if run.gap is not None and run.gap.deck_index == deck_index:
        # the gap is in world x; this run's local x may be mirrored
        lo, hi = run.gap.interval
        gap = (lo, hi) if math.isclose(run.yaw, 0.0) else (-hi, -lo)

    placements: list[Placement] = []
    rejected: dict[tuple[str, str], float] = {}
    x = x_lo
    block = 0
    stalled = 0
    while x < x_hi and stalled <= len(categories):
        cat = categories[block % len(categories)]
        block += 1

        fits = []
        for prod in library[cat]:
            if prod.height <= clearance:
                fits.append(prod)
            else:
                rejected[(prod.category, prod.instance)] = prod.height
        if not fits:
            stalled += 1
            continue

        prod = fits[int(rng.integers(len(fits)))]
        fw, fd = prod.footprint
        slot = fw + st.facing_gap
        n_rows = max(1, min(run.rows_deep,
                            int((y_back - y_front + st.row_gap) // (fd + st.row_gap))))
        n_facings = int(rng.integers(st.min_facings_per_block,
                                     st.max_facings_per_block + 1))

        placed_any = False
        for _ in range(n_facings):
            if gap is not None and x < gap[1] and x + slot > gap[0]:
                x = gap[1]                       # step over the empty run
            if x + slot > x_hi:
                break
            cx = x + slot / 2.0
            for row in range(n_rows):
                cy = y_front + fd / 2.0 + row * (fd + st.row_gap)
                cz = deck_z + prod.height / 2.0 + SEAT_CLEARANCE
                pos = run.to_world((cx, cy, cz))
                rb = cfg.robot
                in_reach = abs(pos[0] - rb.pos_x) <= rb.reach_radius + rb.dynamic_margin
                counter[0] += 1
                placements.append(Placement(
                    product=prod,
                    pos=pos,
                    yaw=prod.yaw_to_face_aisle + run.yaw,
                    dynamic=(row == 0 and run.dynamic_front_row and in_reach),
                    collide=run.collides,
                    run=run.name,
                    deck_index=deck_index,
                    name=f"{run.name}_b{bay}_d{deck_index}_{counter[0]:04d}"
                         f"_{prod.category}_{prod.instance}",
                ))
            x += slot
            placed_any = True
        stalled = 0 if placed_any else stalled + 1

    return placements, rejected


def plan_cage(cfg: SceneConfig, library, rng):
    """Loose stock lying on the cage top tray, all of it free to be picked up."""
    c = cfg.cage
    top = c.top_tray_height + c.tray_thickness / 2.0
    cats = ["cereal", "boxed_food", "milk", "boxed_drink",
            "canned_food", "can", "jam", "yogurt"]
    placements = []
    cols, rows = 4, 2
    for i in range(c.n_loose_products):
        pool = library[cats[i % len(cats)]]
        prod = pool[int(rng.integers(len(pool)))]
        gx = (i % cols - (cols - 1) / 2.0) * (c.length / cols)
        gy = (i // cols - (rows - 1) / 2.0) * (c.width / rows)
        placements.append(Placement(
            product=prod,
            pos=np.array([c.pos_x + gx, c.pos_y + gy,
                          top + prod.height / 2.0 + SEAT_CLEARANCE]),
            yaw=float(rng.uniform(0, 2 * math.pi)),
            dynamic=True,
            collide=True,
            run="cage",
            deck_index=-1,
            name=f"cage_{i:02d}_{prod.category}_{prod.instance}",
        ))
    return placements


# --------------------------------------------------------------------------
# emitters
# --------------------------------------------------------------------------

def emit_product_assets(asset: ET.Element, products: dict[str, Product]) -> None:
    """Declare every mesh/material/texture once, prefixed by product key."""
    for key, p in products.items():
        for t in p.textures:
            E(asset, "texture", name=f"{key}__{t.name}", file=t.file, **t.attrs)
        for m in p.materials:
            attrs = dict(m.attrs)
            if "texture" in attrs:
                attrs["texture"] = f"{key}__{attrs['texture']}"
            E(asset, "material", name=f"{key}__{m.name}", **attrs)
        for mesh in p.meshes:
            E(asset, "mesh", name=f"{key}__{mesh.name}", file=mesh.file,
              scale=mesh.scale, refquat=mesh.refquat)


def emit_product_body(parent: ET.Element, pl: Placement) -> None:
    """One product: full render mesh, plus one primitive collider.

    Keeping the scanned mesh for rendering is the whole point -- this scene is
    for vision data. But those meshes run to six figures of vertices, so they
    are marked contype=0/conaffinity=0 and a box or cylinder fitted to the
    object's measured bounding box does the colliding.
    """
    p = pl.product
    body = E(parent, "body", name=pl.name, pos=pl.pos, quat=yaw_quat(pl.yaw))
    if pl.dynamic:
        E(body, "freejoint")
        E(body, "inertial", pos=fmt(0, 0, 0), mass=p.mass_kg,
          diaginertia=box_inertia(p.mass_kg, p.bbox))

    for vg in p.visual_geoms:
        attrs = dict(type="mesh", mesh=f"{p.key}__{vg.mesh}", group="2",
                     contype="0", conaffinity="0")
        if vg.material:
            attrs["material"] = f"{p.key}__{vg.material}"
        E(body, "geom", **attrs)

    if pl.collide:
        ctype, csize = p.collider
        E(body, "geom", name=f"{pl.name}_coll", type=ctype, size=csize, group="3",
          class_="dyn_stock" if pl.dynamic else "static_stock",
          rgba=fmt(0.5, 0.5, 0.5, 0.0),
          friction=fmt(0.95, 0.3, 0.1), solimp=fmt(0.99, 0.999, 0.001))


def emit_shelf_run(worldbody: ET.Element, cfg: SceneConfig, run: ShelfRun) -> None:
    """Box primitives only. A meshed shelf splits into dozens of collision
    pieces and drags the whole scene down."""
    s = cfg.shelf
    body = E(worldbody, "body", name=f"gondola_{run.name}",
             pos=fmt(run.origin[0], run.origin[1], 0), quat=yaw_quat(run.yaw))
    deck_depth = s.shelf_depth - s.back_panel_thickness

    E(body, "geom", name=f"{run.name}_back_panel", class_="structure", type="box", material="shelf_metal",
      pos=fmt(0, s.shelf_depth - s.back_panel_thickness / 2, s.rack_height / 2),
      size=fmt(s.half_run, s.back_panel_thickness / 2, s.rack_height / 2))

    # one upright at every bay boundary, so n_bays + 1 of them
    for i in range(s.n_bays + 1):
        x = -s.half_run + i * s.bay_length
        x += s.upright_thickness / 2 * (1 if i == 0 else -1 if i == s.n_bays else 0)
        E(body, "geom", name=f"{run.name}_upright_{i}", class_="structure", type="box",
          material="shelf_metal",
          pos=fmt(x, s.shelf_depth / 2, s.rack_height / 2),
          size=fmt(s.upright_thickness / 2, s.shelf_depth / 2, s.rack_height / 2))

    E(body, "geom", name=f"{run.name}_kick_plate", class_="structure", type="box", material="shelf_dark",
      pos=fmt(0, 0.030 + s.board_thickness / 2, s.kick_plate / 2),
      size=fmt(s.half_run, s.board_thickness / 2, s.kick_plate / 2))

    half_bay = s.bay_length / 2 - s.upright_thickness
    for bay in range(s.n_bays):
        cx = -s.half_run + (bay + 0.5) * s.bay_length
        for i, z in enumerate(s.deck_heights):
            tag = f"{run.name}_b{bay}_{i}"
            E(body, "geom", name=f"deck_{tag}", class_="structure", type="box", material="shelf_metal",
              pos=fmt(cx, deck_depth / 2, z - s.board_thickness / 2),
              size=fmt(half_bay, deck_depth / 2, s.board_thickness / 2))
            E(body, "geom", name=f"lip_{tag}", class_="structure", type="box", material="shelf_metal",
              pos=fmt(cx, s.front_lip_thickness / 2, z + s.front_lip / 2),
              size=fmt(half_bay, s.front_lip_thickness / 2, s.front_lip / 2))
            E(body, "geom", name=f"rail_{tag}", class_="structure", type="box", material="price_rail",
              pos=fmt(cx, -s.price_rail_thickness / 2, z - s.board_thickness / 2),
              size=fmt(half_bay, s.price_rail_thickness / 2, s.price_rail_height / 2))


def emit_roll_cage(worldbody: ET.Element, cfg: SceneConfig) -> None:
    """Box primitives and four cylinders. Stands in the aisle, does not roll."""
    c = cfg.cage
    cage = E(worldbody, "body", name="roll_cage", pos=fmt(c.pos_x, c.pos_y, 0))
    hl, hw = c.length / 2, c.width / 2
    up_h = c.top_tray_height - c.base_tray_height

    for name, z in (("base_tray", c.base_tray_height), ("top_tray", c.top_tray_height)):
        E(cage, "geom", name=f"cage_{name}", class_="structure", type="box", material="cage_metal",
          pos=fmt(0, 0, z), size=fmt(hl, hw, c.tray_thickness / 2))

    u = c.upright_section / 2
    for ix, sx in enumerate((-1.0, 1.0)):
        for iy, sy in enumerate((-1.0, 1.0)):
            E(cage, "geom", name=f"cage_upright_{ix}{iy}", class_="structure", type="box",
              material="cage_metal",
              pos=fmt(sx * (hl - u), sy * (hw - u), c.base_tray_height + up_h / 2),
              size=fmt(u, u, up_h / 2))
            E(cage, "geom", name=f"cage_caster_{ix}{iy}", class_="structure", type="cylinder",
              material="cage_dark", euler=fmt(0, math.pi / 2, 0),
              pos=fmt(sx * (hl - 0.07), sy * (hw - 0.07), c.caster_radius),
              size=fmt(c.caster_radius, c.caster_width / 2))

    # open side panels: horizontal bars on the two ends and the back
    bar = c.panel_bar_thickness / 2
    for k in range(c.n_panel_bars):
        z = c.base_tray_height + up_h * (k + 1) / (c.n_panel_bars + 1)
        for ix, sx in enumerate((-1.0, 1.0)):
            E(cage, "geom", name=f"cage_bar_end{ix}_{k}", class_="structure", type="box",
              material="cage_metal", pos=fmt(sx * (hl - u), 0, z),
              size=fmt(bar, hw - u, bar))
        E(cage, "geom", name=f"cage_bar_back_{k}", class_="structure", type="box", material="cage_metal",
          pos=fmt(0, hw - u, z), size=fmt(hl - u, bar, bar))


def emit_robot_base(worldbody: ET.Element, cfg: SceneConfig) -> None:
    """Just a site marking where the lift tower stands.

    The tower itself lives in tower.xml, hand-authored and never generated from
    here: it carries the rotary joint, the prismatic lift, the shoulder bracket
    and the `arm_mount` site. robot.py attaches it at this site.
    """
    rb = cfg.robot
    E(worldbody, "site", name="tower_base_site",
      pos=fmt(rb.pos_x, -rb.standoff, 0.0), size="0.010",
      rgba=fmt(0.95, 0.4, 0.1, 0.35))


def emit_floor(worldbody: ET.Element, asset: ET.Element, cfg: SceneConfig) -> None:
    """Open shop floor only -- no walls, no room.

    A plain pale grey vinyl, the way a supermarket floor actually looks. The
    tile seams are only just visible; a strong checker reads as a placeholder
    swatch rather than a floor.
    """
    f = cfg.floor
    E(asset, "texture", name="sky", type="skybox", builtin="gradient",
      rgb1=fmt(0.93, 0.94, 0.95), rgb2=fmt(0.82, 0.84, 0.86), width="256", height="256")
    E(asset, "texture", name="tile_tex", type="2d", builtin="checker",
      rgb1=fmt(0.855, 0.855, 0.845), rgb2=fmt(0.838, 0.838, 0.828),
      width="512", height="512")
    E(asset, "material", name="floor_tile", texture="tile_tex",
      texrepeat=fmt(30, 30), texuniform="true", specular="0.3", shininess="0.5",
      reflectance="0.04")
    E(asset, "material", name="shelf_metal", rgba=fmt(0.91, 0.91, 0.92, 1),
      specular="0.35", shininess="0.35")
    E(asset, "material", name="shelf_dark", rgba=fmt(0.33, 0.35, 0.38, 1))
    E(asset, "material", name="price_rail", rgba=fmt(0.16, 0.17, 0.19, 1))
    E(asset, "material", name="cage_metal", rgba=fmt(0.58, 0.60, 0.63, 1),
      specular="0.4", shininess="0.4")
    E(asset, "material", name="cage_dark", rgba=fmt(0.18, 0.18, 0.20, 1))

    E(worldbody, "geom", name="floor", type="plane", material="floor_tile",
      class_="structure",
      pos=fmt(0, 0, 0), size=fmt(f.half_extent, f.half_extent, 0.1),
      condim="3", friction=fmt(1.0, 0.005, 0.0001))


def emit_lights(worldbody: ET.Element, cfg: SceneConfig) -> None:
    """Strip lights down the aisle, plus a soft fill so the shelf interiors --
    boxed in on five sides -- are not pitch black."""
    f, s = cfg.floor, cfg.shelf
    aisle_y = -s.aisle_width / 2
    # An overall downward fill. Without it only the pools under the strip
    # lights are lit and the rest of the shop floor reads as a dark void.
    E(worldbody, "light", name="shop_fill", directional="true",
      pos=fmt(0, aisle_y, 6.0), dir=fmt(0, 0, -1), castshadow="false",
      diffuse=fmt(0.45, 0.45, 0.45), specular=fmt(0.02, 0.02, 0.02))
    for i in range(f.n_aisle_lights):
        frac = (i + 0.5) / f.n_aisle_lights
        x = (frac - 0.5) * 2 * (s.half_run + 0.5)
        E(worldbody, "light", name=f"strip_{i}", pos=fmt(x, aisle_y, f.aisle_light_height),
          dir=fmt(0, 0, -1), directional="false",
          castshadow="true" if i == 1 else "false",
          diffuse=fmt(0.26, 0.26, 0.255), specular=fmt(0.07, 0.07, 0.07),
          attenuation=fmt(1, 0.08, 0.02))
    for name, sign in (("fill_near", 1.0), ("fill_far", -1.0)):
        E(worldbody, "light", name=name, pos=fmt(0, aisle_y, 1.55),
          dir=fmt(0, sign, -0.10), directional="false", castshadow="false",
          diffuse=fmt(0.26, 0.26, 0.27), specular=fmt(0.02, 0.02, 0.02))


def emit_cameras(worldbody: ET.Element, cfg: SceneConfig) -> None:
    """Four fixed views.

    `shelf_front` sits above the top of the robot column and looks down past it,
    so the column does not stand in front of the stock it is meant to show.
    """
    s = cfg.shelf
    aisle_y = -s.aisle_width / 2
    near_face = np.array([0.0, 0.20, 1.20])

    eye = np.array([-(s.half_run + 1.45), aisle_y, 1.62])
    E(worldbody, "camera", name="aisle", mode="fixed", fovy="55",
      pos=fmt(eye), xyaxes=look_at(eye, [s.half_run * 0.45, aisle_y, 1.05]))

    eye = np.array([0.0, -s.aisle_width * 0.90, 1.72])
    E(worldbody, "camera", name="shelf_front", mode="fixed", fovy="60",
      pos=fmt(eye), xyaxes=look_at(eye, near_face))

    eye = np.array([0.0, aisle_y, 3.55])
    E(worldbody, "camera", name="overhead", mode="fixed", fovy="62",
      pos=fmt(eye), xyaxes=look_at(eye, eye - [0, 0, 1], up=(0, 1, 0)))

    eye = np.array([-0.32, -s.aisle_width * 0.80, 1.78])
    E(worldbody, "camera", name="over_shoulder", mode="fixed", fovy="56",
      pos=fmt(eye), xyaxes=look_at(eye, [0.10, 0.25, 1.15]))


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

@dataclass
class BuildResult:
    xml: str
    placements: list[Placement]
    rejected: dict
    n_unique_products: int


def build_scene(cfg: SceneConfig, out_path: Path) -> BuildResult:
    rng = np.random.default_rng(cfg.seed)
    library = load_library(PRODUCTS)

    placements: list[Placement] = []
    rejected: dict = {}
    counter = [0]
    for run in cfg.runs:
        if not run.stocked:
            continue
        for slot, deck_index in enumerate(cfg.shelf.stocked_decks):
            for bay in range(cfg.shelf.n_bays):
                pl, rej = plan_bay_deck(cfg, run, bay, deck_index,
                                        run.categories[slot], library, rng, counter)
                placements += pl
                rejected.update(rej)
    placements += plan_cage(cfg, library, rng)

    used: dict[str, Product] = {p.product.key: p.product for p in placements}

    mj = ET.Element("mujoco", model="supermarket_aisle")
    meshdir = "../assets/products"
    E(mj, "compiler", angle="radian", autolimits="true",
      meshdir=meshdir, texturedir=meshdir)
    # elliptic friction cones and a high impratio keep a grasped object from
    # sliding out of the gripper later; implicitfast tolerates the stiff contacts
    E(mj, "option", timestep="0.002", integrator="implicitfast",
      cone="elliptic", impratio="10")
    # Contact masks. Bit 0 = fixed structure, bit 1 = dynamic stock,
    # bit 2 = static stock. Only dynamic stock is allowed to generate contacts:
    # structure never moves, and static back-row stock only has to stop a
    # dynamic item pushed into it. Without this the shelves and the 290 static
    # products contact each other every step for no possible effect.
    default = E(mj, "default")
    E(E(default, "default", class_="structure"), "geom",
      contype="1", conaffinity="2")
    E(E(default, "default", class_="dyn_stock"), "geom",
      contype="2", conaffinity="7")
    E(E(default, "default", class_="static_stock"), "geom",
      contype="4", conaffinity="2")

    visual = E(mj, "visual")
    E(visual, "headlight", ambient=fmt(0.42, 0.42, 0.43),
      diffuse=fmt(0.20, 0.20, 0.20), specular=fmt(0.06, 0.06, 0.06))
    E(visual, "quality", shadowsize="4096", offsamples="8")
    E(visual, "global", offwidth="1920", offheight="1080")
    # znear/zfar are fractions of stat.extent (6.38 m here), NOT metres.
    # znear="0.02" meant a 128 mm near plane, which sliced the wrist and
    # forearm out of the wrist-camera view. 0.003 puts it at ~19 mm.
    # zfar="60" was a 383 m far plane, wasting depth precision on empty space.
    E(visual, "map", znear="0.003", zfar="5")

    asset = E(mj, "asset")
    worldbody = E(mj, "worldbody")
    emit_floor(worldbody, asset, cfg)
    emit_product_assets(asset, used)
    emit_lights(worldbody, cfg)
    emit_cameras(worldbody, cfg)
    for run in cfg.runs:
        emit_shelf_run(worldbody, cfg, run)
    emit_roll_cage(worldbody, cfg)
    emit_robot_base(worldbody, cfg)

    # a free joint is only legal on a direct child of worldbody, so the static
    # stock is grouped under one holder and the dynamic stock is not
    stock = E(worldbody, "body", name="static_stock", pos=fmt(0, 0, 0))
    for pl in placements:
        emit_product_body(worldbody if pl.dynamic else stock, pl)

    ET.indent(mj, space="  ")
    xml = ET.tostring(mj, encoding="unicode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    return BuildResult(xml, placements, rejected, len(used))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "scene.xml")
    ap.add_argument("--view", action="store_true", help="open the result in the viewer")
    args = ap.parse_args()

    cfg = SceneConfig()
    r = build_scene(cfg, args.out)
    s = cfg.shelf

    dyn = sum(p.dynamic for p in r.placements)
    print(f"wrote {args.out}  ({len(r.xml)/1000:.0f} kB)")
    print(f"  gondola run       : {s.n_bays} bays x {s.bay_length:.2f} m = "
          f"{s.run_length:.2f} m, two runs facing across a "
          f"{s.aisle_width:.2f} m aisle")
    print(f"  products placed   : {len(r.placements)}  ({dyn} dynamic / "
          f"{len(r.placements)-dyn} static)")
    print(f"  unique instances  : {r.n_unique_products}")
    for run in cfg.runs:
        n = sum(1 for p in r.placements if p.run == run.name)
        kind = "dynamic" if run.dynamic_front_row else "static"
        print(f"    run {run.name:5s}       : {n} products, "
              f"{run.rows_deep} rows deep, front row {kind}")
    n_cage = sum(1 for p in r.placements if p.run == "cage")
    print(f"    cage            : {n_cage} products")
    gap = cfg.runs[0].gap
    lo, hi = gap.interval
    print(f"  gap               : deck {gap.deck_index} "
          f"(z={s.deck_heights[gap.deck_index]:.2f}), "
          f"x {lo:+.2f}..{hi:+.2f} ({gap.width*1000:.0f} mm)")
    print(f"  rejected too tall : {len(r.rejected)}")

    if args.view:
        import mujoco
        import mujoco.viewer
        mujoco.viewer.launch(mujoco.MjModel.from_xml_path(str(args.out)))


if __name__ == "__main__":
    main()
