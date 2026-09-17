"""Bring over-detailed product scans down to a sensible triangle budget.

    python tools/simplify_meshes.py              # report only, changes nothing
    python tools/simplify_meshes.py --apply      # rewrite meshes over budget

Why this exists
---------------
Two of the scanned products shipped with absurd detail: `canned_food_9` at
197,956 triangles and `yogurt_3` at 194,470, for objects about 7 cm across. With
30 copies between them on the shelves they were 79% of the 7.49 million
triangles drawn every frame, and a single aisle camera took ~700 ms to render on
this machine's Intel UHD 630 -- which made the aisle unusable for anything
interactive, teleoperation included. Hiding just those two took a render from
722 ms to 165 ms.

Only VISUAL meshes are touched. Collision meshes are separate files under
`collision/` and are left exactly as they are, so contacts, grasping and
physics do not change. The visual geom does carry a density, so its volume
feeds the product's mass; after every rewrite the density is rescaled exactly as
`build_products` does, so each product still weighs its specified target.

Why 4,000 triangles
-------------------
From `tower_eye`, 260 mm off the shelf, a 7 cm can is ~85 px across at 640x480.
A cylinder at that size is visually round with a few dozen sides. 4,000
triangles leaves a wide margin for the wrist cameras, which get much closer.

Safety
------
Nothing is written until the simplified mesh has been rendered next to the
original, from four sides, and compared. Loading cleanly and keeping its mass
proves nothing about how a mesh looks -- a mesh with every texture coordinate
collapsed still loads and weighs exactly the same. The check was tested against
exactly that: corrupted texture coordinates score a mean pixel difference of
22.4 and are rejected; a good simplification of the same can scores 2.6.

The OBJ is written here rather than by MeshLab's exporter, which stores one
texture coordinate per vertex. A vertex on a label seam has one position and
two texture coordinates, so that merges the two sides of the seam; writing per
face corner keeps them apart.

The automatic check is necessary but NOT sufficient, and was shown not to be:
it passed a one-pixel seam line on one can and a warped label on one box, both
caught only by looking at `--sheet`. Always look at the sheet before committing.

The originals are in git; `git checkout -- assets/products` undoes all of it.

Needs `pymeshlab` (texture-aware quadric decimation), which is a build-time tool
only -- nothing at runtime imports it, so it is not in requirements.txt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRODUCTS = REPO / "assets" / "products"

DEFAULT_BUDGET = 4000
# Leave anything at or below this alone. A mesh at 5-10k triangles saves almost
# nothing when simplified and still carries the risk of a visible defect -- and
# the automatic check cannot rule those out: a one-pixel line down a label seam
# on canned_food_18 (5,244 triangles) scored a mean pixel difference of 4.0 and
# passed. Only the meshes where the saving is large are worth that risk, and
# every one of those gets looked at.
ONLY_OVER = 12000
# Simplified, checked by eye, and rejected. Printed text on a flat face is where
# decimation slides the texture: boxed_food_4's "MAC & CHEESE" came out warped
# while scoring within the automatic limits.
DO_NOT_SIMPLIFY = {
    "boxed_food/boxed_food_4/visual/model_normalized_0.obj",
}
MAX_MASS_CHANGE = 0.01          # fraction
MAX_PIXEL_DIFF = 10.0           # mean abs difference over the object, 0..255


def count_faces(obj: Path) -> int:
    n = 0
    with obj.open("rb") as f:
        for line in f:
            if line.startswith(b"f "):
                n += 1
    return n


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray,
              wedge_uv: np.ndarray, normals: np.ndarray | None) -> None:
    """OBJ with separate position and texture indices per face corner.

    Texture coordinates live on face corners ("wedges"), not on vertices: a
    vertex on a label seam has one position and two UVs. MeshLab's own exporter
    writes one UV per vertex, which merges the two sides of every seam.
    """
    uv = np.round(wedge_uv.reshape(-1, 2), 7)
    uniq, inv = np.unique(uv, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 3)
    lines = ["# simplified by tools/simplify_meshes.py"]
    lines += [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in verts]
    lines += [f"vt {u:.7f} {v:.7f}" for u, v in uniq]
    if normals is not None:
        lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
    for (a, b, c), (ta, tb, tc) in zip(faces + 1, inv + 1):
        if normals is not None:
            lines.append(f"f {a}/{ta}/{a} {b}/{tb}/{b} {c}/{tc}/{c}")
        else:
            lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def simplify(obj: Path, budget: int) -> None:
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj))
    # The scans duplicate every vertex on a texture seam. Weld positions first,
    # or those seams become boundaries the decimator is not allowed to collapse;
    # the UVs survive the weld because they are stored per corner.
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=budget, qualitythr=0.3, extratcoordw=1.0,
        preserveboundary=True, boundaryweight=1.0, optimalplacement=True,
        preservenormal=True, planarquadric=True)
    ms.compute_normal_per_vertex()
    m = ms.current_mesh()
    if not m.has_wedge_tex_coord():
        raise RuntimeError("texture coordinates lost during decimation")
    write_obj(obj, m.vertex_matrix(), m.face_matrix(),
              m.wedge_tex_coord_matrix(), m.vertex_normal_matrix())


def restore_mass(product: Path) -> float:
    """Put the product's mass back on its target, exactly, the way the build does.

    The visual geom carries a density, so a simplified mesh with a fraction of a
    percent less volume weighs a fraction of a percent less. `verify_products`
    holds every instance to its target within 0.1 g, and it is right to: the
    masses are specified, not incidental. So this applies the same single-factor
    density rescale `build_products` uses when it first builds the product,
    rather than inventing a second way to set mass.
    """
    import xml.etree.ElementTree as ET
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_products import patch_density, total_mass    # noqa: E402
    from product_spec import PRODUCTS as SPECS              # noqa: E402

    category = product.parent.name
    target = next(sp.mass_kg for sp in SPECS if sp.category == category)
    model_xml = product / "model.xml"
    tree = ET.parse(model_xml)
    patch_density(tree.getroot(), target / total_mass(model_xml))
    tree.write(model_xml, encoding="utf-8", xml_declaration=True)
    return total_mass(model_xml) - target


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def _render(model_xml: Path):
    spec = mujoco.MjSpec.from_file(str(model_xml))
    spec.worldbody.add_light(pos=[0.3, -0.3, 0.6], dir=[-0.4, 0.4, -1.0],
                             diffuse=[0.8, 0.8, 0.8])
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 480, 360
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = next(i for i in range(m.nbody)
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) == "object")
    # frame the object whatever its size
    r = float(max(m.geom_rbound[m.body_geomadr[body]:
                                m.body_geomadr[body] + m.body_geomnum[body]]))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.xipos[body]
    cam.distance = max(2.2 * r, 0.08)
    cam.elevation = -15.0
    views = []
    # One renderer, closed before the next opens: a second GL context opened
    # while the first is live renders black on Windows, which once made a good
    # mesh look broken. The background stays black for the object mask.
    with mujoco.Renderer(m, 360, 480) as rend:
        rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        for az in (0.0, 90.0, 180.0, 270.0):
            cam.azimuth = az
            rend.update_scene(d, camera=cam)
            views.append(rend.render().copy())
    return float(mujoco.mj_getTotalmass(m)), np.hstack(views)


def compare(orig_xml: Path, new_xml: Path) -> dict:
    mass0, img0 = _render(orig_xml)
    mass1, img1 = _render(new_xml)
    mask = (img0.max(axis=2) > 8) | (img1.max(axis=2) > 8)
    diff = np.abs(img0.astype(int) - img1.astype(int)).max(axis=2)
    return dict(mass0=mass0, mass1=mass1,
                mass_change=(mass1 - mass0) / mass0,
                pixel_diff=float(diff[mask].mean()) if mask.any() else 0.0,
                img0=img0, img1=img1)


# --------------------------------------------------------------------------
# one mesh, in its own process
# --------------------------------------------------------------------------
def process_one(obj: Path, budget: int, write: bool, image_out: Path | None) -> dict:
    """Simplify a staged copy, verify it, and copy it over the original if it passes."""
    product = obj.parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / product.name
        shutil.copytree(product, staged)
        staged_obj = staged / "visual" / obj.name
        simplify(staged_obj, budget)
        res = compare(product / "model.xml", staged / "model.xml")
        out = dict(faces=count_faces(staged_obj),
                   size=staged_obj.stat().st_size,
                   mass_change=res["mass_change"],
                   pixel_diff=res["pixel_diff"])
        out["ok"] = (abs(res["mass_change"]) <= MAX_MASS_CHANGE
                     and res["pixel_diff"] <= MAX_PIXEL_DIFF)
        if out["ok"] and image_out is not None:
            import cv2
            pair = cv2.cvtColor(np.vstack([res["img0"], res["img1"]]), cv2.COLOR_RGB2BGR)
            label = f"{obj.relative_to(PRODUCTS)}  {count_faces(obj):,} -> {out['faces']:,}"
            cv2.putText(pair, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (60, 220, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(image_out), pair)
        if out["ok"] and write:
            shutil.copyfile(staged_obj, obj)
            out["mass_residual"] = restore_mass(product)
    return out


def run_isolated(obj: Path, budget: int, write: bool, image_out: Path | None,
                 timeout: float = 600.0) -> dict:
    """Run process_one in a child process.

    pymeshlab can take the whole interpreter down with a segfault -- it did, on
    a mesh whose decimation had already failed cleanly -- and a crash half way
    through a batch leaves some products rewritten and the rest not. One process
    per mesh makes a crash cost that one mesh, and gives every render a fresh GL
    context as a side effect.
    """
    import json
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(obj),
           "--budget", str(budget)]
    if write:
        cmd.append("--apply")
    if image_out is not None:
        cmd += ["--image", str(image_out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(ok=False, error=f"timed out after {timeout:.0f} s")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    err = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return dict(ok=False, error=f"exit {proc.returncode}: {err}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--only-over", type=int, default=ONLY_OVER,
                    help="leave meshes at or below this many triangles alone")
    ap.add_argument("--apply", action="store_true", help="rewrite meshes over budget")
    ap.add_argument("--sheet", type=Path, default=None,
                    help="write a before/after image of every simplified mesh")
    ap.add_argument("--one", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--image", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one is not None:                      # child process
        import json
        try:
            res = process_one(args.one, args.budget, args.apply, args.image)
        except Exception as exc:
            res = dict(ok=False, error=str(exc).splitlines()[-1])
        print("RESULT " + json.dumps(res), flush=True)
        return 0

    over = [(obj, count_faces(obj)) for obj in sorted(PRODUCTS.glob("*/*/visual/*.obj"))]
    over = [(o, n) for o, n in over
            if n > max(args.budget, args.only_over) and o.relative_to(PRODUCTS).as_posix()
            not in DO_NOT_SIMPLIFY]
    print(f"{len(over)} visual meshes over {args.only_over:,} triangles "
          f"(target {args.budget:,}):")
    for obj, n in over:
        print(f"  {obj.relative_to(PRODUCTS)}  {n:,}  ({obj.stat().st_size/1e6:.1f} MB)")
    if not args.apply:
        print("\nreport only; pass --apply to rewrite them")
        return 0

    print()
    done, skipped, images = [], [], []
    img_dir = Path(tempfile.mkdtemp())
    for k, (obj, n) in enumerate(over):
        name = obj.relative_to(PRODUCTS)
        # A mesh that fails the visual check at the budget gets a bigger one,
        # rather than being written worse or being left at 100k triangles.
        budget, result = args.budget, None
        while budget < n:
            image = img_dir / f"{k:03d}.png"
            result = run_isolated(obj, budget, True, image if args.sheet else None)
            if result.get("ok") or "error" in result:
                break
            print(f"  ...  {name}: pixel diff {result['pixel_diff']:.1f} at "
                  f"{budget:,}, retrying at {budget*2:,}")
            budget *= 2
        if result and result.get("ok"):
            print(f"  ok   {name}  {n:,} -> {result['faces']:,} tris, "
                  f"{result['size']/1e6:.2f} MB, mass {100*result['mass_change']:+.2f}%, "
                  f"pixel diff {result['pixel_diff']:.1f}")
            done.append(name)
            if args.sheet and (img_dir / f"{k:03d}.png").exists():
                images.append(img_dir / f"{k:03d}.png")
        else:
            why = (result or {}).get("error") or "no budget below the original passed"
            print(f"  skip {name}: {why}")
            skipped.append((name, why))

    if args.sheet and images:
        import cv2
        sheet = np.vstack([cv2.imread(str(p)) for p in images])
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.sheet), sheet)
        print(f"\nwrote {args.sheet}")
    shutil.rmtree(img_dir, ignore_errors=True)

    print(f"\n{len(done)} rewritten and verified, {len(skipped)} left unchanged")
    for name, why in skipped:
        print(f"  - {name}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
