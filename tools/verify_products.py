"""Verify every extracted product loads, and report geometry vs assigned mass."""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from product_spec import PRODUCTS      # noqa: E402

OUT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "products"


def bbox_of(model_xml: Path):
    root = ET.parse(model_xml).getroot()
    for geom in root.iter("geom"):
        if geom.get("name") == "reg_bbox":
            return np.array([2 * float(v) for v in geom.get("size").split()])
    raise ValueError(f"{model_xml}: no reg_bbox")


def main():
    print(f"{'category':13s} {'family':14s} {'n':>3s} {'geoms':>5s} "
          f"{'W':>5s} {'D':>5s} {'H':>5s}  {'shapeVol':>8s} "
          f"{'mass':>6s} {'rho':>6s} {'band':>12s} {'':4s}")
    print("-" * 104)
    grand_files = grand_bytes = 0
    failures = []
    for spec in PRODUCTS:
        cat_dir = OUT_ROOT / spec.category
        insts = sorted(p for p in cat_dir.iterdir() if p.is_dir())
        boxes, ngeoms, n_ok = [], [], 0
        for inst in insts:
            xml = inst / "model.xml"
            model = mujoco.MjModel.from_xml_path(str(xml))   # raises if broken
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all():
                raise ValueError(f"{xml}: non-finite qpos")
            total = float(mujoco.mj_getTotalmass(model))
            if abs(total - spec.mass_kg) > 1e-4:
                raise ValueError(f"{xml}: mass {total} != target {spec.mass_kg}")
            n_ok += 1
            ngeoms.append(model.ngeom)
            boxes.append(bbox_of(xml))

        b = np.array(boxes)
        med = np.median(b, axis=0)
        bbox_vol = float(np.prod(med))
        fill = math.pi / 4 if spec.is_cylinder else 1.0
        shape_vol = bbox_vol * fill
        rho = spec.mass_kg / shape_vol
        lo, hi = spec.band
        ok = lo <= rho <= hi
        if not ok:
            failures.append(f"{spec.category}: rho {rho:.0f} outside {lo:.0f}-{hi:.0f}")
        print(f"{spec.category:13s} {spec.family:14s} {len(insts):3d} "
              f"{int(np.median(ngeoms)):5d} "
              f"{med[0]*1000:5.0f} {med[1]*1000:5.0f} {med[2]*1000:5.0f}  "
              f"{shape_vol*1e6:7.0f}ml "
              f"{spec.mass_kg:6.2f} {rho:6.0f} {lo:5.0f}-{hi:<5.0f} "
              f"{'ok' if ok else 'FLAG':>4s}")
        for p in cat_dir.rglob("*"):
            if p.is_file():
                grand_files += 1
                grand_bytes += p.stat().st_size
    print("-" * 104)
    print(f"total: {grand_files} files, {grand_bytes/1e6:.1f} MB on disk")
    print(f"all {sum(1 for _ in PRODUCTS)} categories loaded, "
          f"every instance mass matched its target")
    print("rho = assigned mass / shape volume, kg/m^3.  Water = 1000.")
    if failures:
        raise SystemExit("density band violations: " + "; ".join(failures))


if __name__ == "__main__":
    main()
