"""Extract the eight product categories from the RoboCasa objaverse pack.

Produces assets/products/<category>/<instance>/ holding a self-contained MJCF:
mesh paths stay relative, RoboCasa's per-category scale is baked in, and the
density is rescaled so the object's total mass matches the value in
product_spec.py rather than RoboCasa's uniform density=100 kg/m^3.
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_spec import PRODUCTS                      # noqa: E402
from robocasa_registry import load_categories          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "assets" / "products"
PACK = "objaverse"


def as_xyz(scale):
    """RoboCasa scale is either a scalar or a per-axis triple."""
    if scale is None:
        return (1.0, 1.0, 1.0)
    if isinstance(scale, (int, float)):
        return (float(scale),) * 3
    return tuple(float(v) for v in scale)


def instances_in_zip(zf, category):
    """{instance_name: [member paths]} for <pack>/<category>/<instance>/..."""
    found = {}
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) < 4 or parts[-1] == "":
            continue
        if parts[0] == PACK and parts[1] == category:
            found.setdefault(parts[2], []).append(name)
    return {k: v for k, v in found.items() if any(m.endswith("model.xml") for m in v)}


def total_mass(model_xml: Path) -> float:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    return float(mujoco.mj_getTotalmass(model))


def bounding_box(model_xml: Path):
    """Axis-aligned bbox (x, y, z) in metres.

    Read off the `reg_bbox` geom RoboCasa ships with every instance -- it is the
    authored extent of the object, and it is what we size shelf facings from.
    """
    root = ET.parse(model_xml).getroot()
    for geom in root.iter("geom"):
        if geom.get("name") == "reg_bbox":
            half = [float(v) for v in geom.get("size").split()]
            return tuple(2.0 * h for h in half)
    raise ValueError(f"{model_xml}: no reg_bbox geom")


def patch_scale(root: ET.Element, scale) -> None:
    """Apply RoboCasa's per-category scale on top of what model.xml already has.

    The objaverse MJCFs already carry a per-instance mesh scale that brings the
    normalised mesh to roughly real size. RoboCasa's category scale multiplies
    that (see robocasa/models/objects/objects.py, `m_scale *= self._scale`), so
    we multiply too -- overwriting would discard the instance's own sizing.
    """
    sx, sy, sz = scale
    if (sx, sy, sz) == (1.0, 1.0, 1.0):
        return
    for mesh in root.iter("mesh"):
        have = [float(v) for v in mesh.get("scale", "1 1 1").split()]
        if len(have) == 1:
            have = have * 3
        mesh.set("scale", f"{have[0]*sx} {have[1]*sy} {have[2]*sz}")
    for geom in root.iter("geom"):
        if geom.get("type") == "box" and geom.get("size"):
            a, b, c = (float(v) for v in geom.get("size").split())
            geom.set("size", f"{a*sx} {b*sy} {c*sz}")
        if geom.get("pos"):
            a, b, c = (float(v) for v in geom.get("pos").split())
            geom.set("pos", f"{a*sx} {b*sy} {c*sz}")
    for site in root.iter("site"):
        if site.get("pos"):
            a, b, c = (float(v) for v in site.get("pos").split())
            site.set("pos", f"{a*sx} {b*sy} {c*sz}")


def neutralise_region_mass(root: ET.Element) -> None:
    """Make the `reg_bbox` marker weightless.

    It is a non-colliding annotation (contype=0, conaffinity=0, transparent),
    but RoboCasa gives it mass: some instances set a density on the
    <default class="region"> block, others set none and so inherit MuJoCo's
    default 1000 kg/m^3. Either way it inflates the object's mass, and the
    inherited case is invisible to a density rescale. Pin it to zero instead.
    MuJoCo rejects a geom carrying both mass and density, so density goes.
    """
    for default in root.iter("default"):
        if default.get("class") != "region":
            continue
        for geom in default.findall("geom"):
            geom.attrib.pop("density", None)
            geom.set("mass", "0")
    for geom in root.iter("geom"):
        if geom.get("class") == "region":
            geom.attrib.pop("density", None)


def patch_density(root: ET.Element, factor: float) -> int:
    """Scale every density in the file by one factor.

    Mass is linear in density, so this hits the target mass exactly while
    preserving how inertia is distributed over the shape.

    Two authoring styles appear in the pack: some geoms carry `density`
    directly, others carry `class="collision"` and inherit it from a
    <default> block. Both are covered by scaling the defaults as well as the
    geoms. A geom that resolves to neither would silently inherit MuJoCo's
    built-in 1000 kg/m^3 and escape the rescale, so that is an error.
    """
    # class name -> the <geom> prototype in that <default> block ("" = root)
    proto = {}
    for d in root.iter("default"):
        for g in d.findall("geom"):
            proto[d.get("class") or ""] = g

    n = 0
    for g in proto.values():
        if g.get("density") is not None:
            g.set("density", f"{float(g.get('density')) * factor:.9f}")
            n += 1

    worldbody = root.find("worldbody")
    for geom in worldbody.iter("geom"):
        if geom.get("density") is not None:
            geom.set("density", f"{float(geom.get('density')) * factor:.9f}")
            n += 1
            continue
        if geom.get("mass") is not None:
            continue
        # no own density: it must come from its class, or the root default
        chain = [geom.get("class") or "", ""]
        provider = next((proto[c] for c in chain if c in proto), None)
        if provider is None or (
            provider.get("density") is None and provider.get("mass") is None
        ):
            raise ValueError(
                f"geom {geom.get('name') or geom.get('mesh')!r} resolves to no "
                "density or mass; it would inherit MuJoCo's default 1000 kg/m^3"
            )
    return n


def check_paths_relative(root: ET.Element, where: str) -> None:
    for tag in ("mesh", "texture"):
        for el in root.iter(tag):
            f = el.get("file")
            if f and (Path(f).is_absolute() or f.startswith("..")):
                raise ValueError(f"{where}: non-relative asset path {f!r}")


def build(zip_path: Path, limit_per_category: int | None):
    zf = zipfile.ZipFile(zip_path)
    registry = load_categories()
    rows = []

    for spec in PRODUCTS:
        info = registry[spec.category]
        excluded = set(info["exclude"].get(PACK, []))
        scale = as_xyz(info["scale"].get(PACK))

        found = instances_in_zip(zf, spec.category)
        dropped = sorted(n for n in found if n in excluded)
        kept = sorted(n for n in found if n not in excluded)
        if limit_per_category:
            kept = kept[:limit_per_category]

        cat_dir = OUT_ROOT / spec.category
        if cat_dir.exists():
            shutil.rmtree(cat_dir)
        cat_dir.mkdir(parents=True)

        per_instance = []
        for inst in kept:
            for member in found[inst]:
                rel = "/".join(member.split("/")[3:])
                if not rel:
                    continue
                dest = cat_dir / inst / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            model_xml = cat_dir / inst / "model.xml"
            tree = ET.parse(model_xml)
            root = tree.getroot()
            check_paths_relative(root, f"{spec.category}/{inst}")
            patch_scale(root, scale)
            neutralise_region_mass(root)
            tree.write(model_xml, encoding="utf-8", xml_declaration=True)

            mass_before = total_mass(model_xml)          # RoboCasa density=100
            factor = spec.mass_kg / mass_before
            patch_density(root, factor)
            tree.write(model_xml, encoding="utf-8", xml_declaration=True)

            mass_after = total_mass(model_xml)
            bbox = bounding_box(model_xml)
            per_instance.append((inst, mass_before, mass_after, bbox))

        rows.append((spec, len(found), len(dropped), len(kept), scale, per_instance))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", type=Path)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap instances per category (for a quick smoke run)")
    args = ap.parse_args()

    rows = build(args.zip_path, args.limit)

    print(f"\n{'category':13s} {'inZip':>5s} {'excl':>4s} {'kept':>4s} {'scale':>5s} "
          f"{'bbox W x D x H (mm)':>24s} {'massBefore':>10s} {'massAfter':>9s}")
    print("-" * 92)
    total_files = total_bytes = 0
    for spec, n_zip, n_excl, n_kept, scale, insts in rows:
        masses_before = [m for _, m, _, _ in insts]
        masses_after = [m for _, _, m, _ in insts]
        bb = [b for _, _, _, b in insts]
        mean_bb = tuple(sum(b[i] for b in bb) / len(bb) * 1000 for i in range(3)) if bb else (0, 0, 0)
        print(f"{spec.category:13s} {n_zip:5d} {n_excl:4d} {n_kept:4d} {scale[0]:5.2f} "
              f"{mean_bb[0]:7.0f} x{mean_bb[1]:6.0f} x{mean_bb[2]:6.0f}   "
              f"{sum(masses_before)/len(masses_before):10.3f} "
              f"{sum(masses_after)/len(masses_after):9.4f}")
        for p in (OUT_ROOT / spec.category).rglob("*"):
            if p.is_file():
                total_files += 1
                total_bytes += p.stat().st_size
    print("-" * 92)
    print(f"total on disk: {total_files} files, {total_bytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()
