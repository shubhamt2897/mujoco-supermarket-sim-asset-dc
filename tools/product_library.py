"""Read the extracted product MJCFs so scene.py can emit them as shared assets.

Attaching each product with MjSpec duplicates its mesh assets once per copy --
80 products came to 895 meshes and 1.6x realtime. A shelf bay needs far more
than 80 facings, so the scene instead declares every mesh, material and texture
once and lets each placement reference them.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRODUCTS_ROOT = REPO / "assets" / "products"


@dataclass(frozen=True)
class Mesh:
    name: str
    file: str            # relative to PRODUCTS_ROOT
    scale: str | None
    refquat: str | None


@dataclass(frozen=True)
class Texture:
    name: str
    file: str
    attrs: dict


@dataclass(frozen=True)
class Material:
    name: str
    attrs: dict          # includes 'texture' when it has one


@dataclass(frozen=True)
class VisualGeom:
    mesh: str
    material: str | None


@dataclass
class Product:
    """One product instance: its render assets and its measured extent."""
    category: str
    instance: str
    family: str                           # key into product_spec.DENSITY_BANDS
    bbox: np.ndarray                      # full extents (x, y, z) in metres
    mass_kg: float
    meshes: list[Mesh] = field(default_factory=list)
    textures: list[Texture] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    visual_geoms: list[VisualGeom] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.category}__{self.instance}"

    @property
    def collider(self) -> tuple[str, list[float]]:
        """A primitive standing in for the render mesh during collision.

        The render meshes are scans: canned_food_9 carries 175k vertices and
        yogurt_3 116k. Colliding those through their convex hull dropped the
        scene to 1.9x realtime. A primitive fitted to the measured bounding box
        costs almost nothing, and for a can standing upright it *is* the hull.

        A cylinder is only used when the footprint is actually round. Not every
        member of the cylinder family stands up -- canned_food_9 measures
        60 x 105 x 55 mm, a tin lying on its side -- and wrapping that in a
        105 mm cylinder makes it collide far outside its own bounding box.
        """
        x, y, z = (float(v) for v in self.bbox)
        if self.family == "cylinder" and abs(x - y) <= 0.12 * max(x, y):
            return "cylinder", [(x + y) / 4.0, z / 2.0]
        return "box", [x / 2.0, y / 2.0, z / 2.0]

    @property
    def footprint(self) -> tuple[float, float]:
        """(along-bay, into-shelf) extent, taken from the collider.

        Derived from the collider rather than the raw bbox so that facing and
        row spacing can never disagree with what actually collides.
        """
        kind, size = self.collider
        if kind == "cylinder":
            d = 2.0 * size[0]
            return (d, d)
        w, depth = 2.0 * size[0], 2.0 * size[1]
        return (max(w, depth), min(w, depth))

    @property
    def yaw_to_face_aisle(self) -> float:
        """Shops turn a pack so its wide face meets the shopper. Radians."""
        if self.collider[0] == "cylinder":
            return 0.0
        return np.pi / 2 if self.bbox[1] > self.bbox[0] else 0.0

    @property
    def height(self) -> float:
        return float(self.bbox[2])


def _resolve(geom: ET.Element, protos: dict[str, ET.Element], attr: str) -> str | None:
    """Attribute value on the geom, else inherited from its <default> class."""
    if geom.get(attr) is not None:
        return geom.get(attr)
    for cls in (geom.get("class") or "", ""):
        proto = protos.get(cls)
        if proto is not None and proto.get(attr) is not None:
            return proto.get(attr)
    return None


def load_product(category: str, instance: str, family: str, mass_kg: float) -> Product:
    model_xml = PRODUCTS_ROOT / category / instance / "model.xml"
    root = ET.parse(model_xml).getroot()
    rel_dir = f"{category}/{instance}"

    protos: dict[str, ET.Element] = {}
    for d in root.iter("default"):
        for g in d.findall("geom"):
            protos[d.get("class") or ""] = g

    product = Product(category=category, instance=instance, family=family,
                      bbox=np.zeros(3), mass_kg=mass_kg)

    for el in root.iter("mesh"):
        product.meshes.append(Mesh(
            name=el.get("name"),
            file=f"{rel_dir}/{el.get('file')}".replace("\\", "/"),
            scale=el.get("scale"),
            refquat=el.get("refquat"),
        ))
    for el in root.iter("texture"):
        attrs = {k: v for k, v in el.attrib.items() if k not in ("name", "file")}
        product.textures.append(Texture(
            name=el.get("name"),
            file=f"{rel_dir}/{el.get('file')}".replace("\\", "/"),
            attrs=attrs,
        ))
    for el in root.iter("material"):
        attrs = {k: v for k, v in el.attrib.items() if k != "name"}
        product.materials.append(Material(name=el.get("name"), attrs=attrs))

    worldbody = root.find("worldbody")
    bbox = None
    for geom in worldbody.iter("geom"):
        if geom.get("name") == "reg_bbox":
            bbox = 2.0 * np.array([float(v) for v in geom.get("size").split()])
            continue
        # a geom that collides with nothing is there to be looked at
        if _resolve(geom, protos, "contype") == "0" and \
           _resolve(geom, protos, "conaffinity") == "0":
            product.visual_geoms.append(
                VisualGeom(mesh=geom.get("mesh"), material=geom.get("material"))
            )
    if bbox is None:
        raise ValueError(f"{model_xml}: no reg_bbox geom")
    if not product.visual_geoms:
        raise ValueError(f"{model_xml}: no visual geoms found")
    product.bbox = bbox

    # keep only the meshes the visual geoms actually reference
    used = {vg.mesh for vg in product.visual_geoms}
    product.meshes = [m for m in product.meshes if m.name in used]
    return product


def load_library(specs, per_category: int | None = None) -> dict[str, list[Product]]:
    """category -> [Product], in sorted instance order."""
    library = {}
    for spec in specs:
        cat_dir = PRODUCTS_ROOT / spec.category
        instances = sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
        if per_category:
            instances = instances[:per_category]
        library[spec.category] = [
            load_product(spec.category, inst, spec.family, spec.mass_kg)
            for inst in instances
        ]
    return library
