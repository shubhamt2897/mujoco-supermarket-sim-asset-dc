"""Single source of truth for which RoboCasa categories we use and what they weigh.

RoboCasa ships every object with density=100 kg/m^3 (a tenth of water). That is
in the right region for a cereal carton and badly wrong for a can or a full jar,
which are near 1000. We therefore override mass explicitly, per category.

The masses below are sized to the meshes we actually have, which are smaller
than the standard retail pack the name suggests -- the `milk` mesh is a 525 ml
carton, not a litre. Each mass is measured shape volume x a plausible gross
density for that pack (contents plus packaging), so glass and steel sit above
1000 while a box of cereal, which is mostly air, sits near 150. Volumes come
from tools/verify_products.py, which also enforces the bands below.
"""

from dataclasses import dataclass

# Plausible gross density (kg/m^3) per shape family: contents + packaging,
# measured over the whole bounding shape including void. Anything outside its
# band is a sizing or mass error, not a product we actually stock.
DENSITY_BANDS: dict[str, tuple[float, float]] = {
    # Dry goods in card: cereal, pasta, crackers. Mostly air, and the box is
    # deliberately larger than its contents. Nothing in this family should
    # approach water -- above ~700 means the mass or the mesh is wrong.
    "dry_carton":    (80.0, 700.0),
    # Liquid in a card/foil carton: milk, juice. Contents are close to water
    # and the carton is filled, so these sit just above 1000.
    "liquid_carton": (900.0, 1250.0),
    # Steel can, glass jar, plastic pot. Packaging is heavy relative to volume,
    # so these legitimately exceed water, but not by half again.
    "cylinder":      (900.0, 1500.0),
}


@dataclass(frozen=True)
class ProductSpec:
    category: str      # RoboCasa category name
    family: str        # key into DENSITY_BANDS
    mass_kg: float     # corrected mass, overriding RoboCasa's density=100
    note: str          # measured volume and the gross density it implies

    @property
    def band(self) -> tuple[float, float]:
        return DENSITY_BANDS[self.family]

    @property
    def is_cylinder(self) -> bool:
        """Cylinders fill pi/4 of their bounding box; cartons fill all of it."""
        return self.family == "cylinder"


# The eight shelf-stable categories we stock the bay with. All are graspable=True
# in robocasa/models/objects/kitchen_objects.py and all live in the objaverse pack.
PRODUCTS: tuple[ProductSpec, ...] = (
    ProductSpec("boxed_food",  "dry_carton",    0.35, "834 ml dry goods carton @ 420"),
    ProductSpec("cereal",      "dry_carton",    0.09, "593 ml cereal box @ 152"),
    ProductSpec("boxed_drink", "liquid_carton", 0.25, "213 ml drink carton @ 1175"),
    ProductSpec("milk",        "liquid_carton", 0.54, "525 ml carton @ 1029"),
    ProductSpec("canned_food", "cylinder",      0.31, "240 ml steel can @ 1290"),
    ProductSpec("can",         "cylinder",      0.35, "301 ml drink can @ 1163"),
    ProductSpec("jam",         "cylinder",      0.28, "203 ml glass jar @ 1381"),
    ProductSpec("yogurt",      "cylinder",      0.20, "186 ml plastic pot @ 1074"),
)

BY_CATEGORY = {p.category: p for p in PRODUCTS}
