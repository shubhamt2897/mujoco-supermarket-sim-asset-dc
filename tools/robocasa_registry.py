"""Read RoboCasa's OBJ_CATEGORIES without importing robocasa.

Importing robocasa pulls in robosuite, and `pip install -e robocasa` pins
mujoco==3.3.1 and drags in torch via lerobot/tianshou. We only need a few
declarative fields, so we parse the source with `ast` instead.
"""

import ast
from pathlib import Path

KITCHEN_OBJECTS = (
    Path(__file__).resolve().parents[1]
    / "robocasa" / "robocasa" / "models" / "objects" / "kitchen_objects.py"
)

# Fields we lift out of each per-pack dict.
_PACKS = ("aigen", "objaverse", "lightwheel")


def _value(node):
    """Evaluate a declarative expression node.

    The exclude lists are not all plain literals -- some are `[...] + [...]`,
    some are comprehensions, one is `list(set(...) - set(...))`. They are all
    closed expressions over literals, so compiling and evaluating the node is
    both correct and sufficient.
    """
    try:
        return ast.literal_eval(node)
    except ValueError:
        return eval(compile(ast.Expression(node), "<kitchen_objects>", "eval"), {}, {})


def load_categories(path: Path = KITCHEN_OBJECTS) -> dict:
    """category name -> {graspable, types, packs, exclude, scale, model_folders}"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "OBJ_CATEGORIES" for t in node.targets)
        ):
            continue
        cats = {}
        for kw in node.value.keywords:
            info = {
                "graspable": None, "types": None, "packs": [],
                "exclude": {}, "scale": {}, "model_folders": {},
            }
            for sub in kw.value.keywords:
                if sub.arg in _PACKS:
                    info["packs"].append(sub.arg)
                    for s2 in sub.value.keywords:
                        if s2.arg in ("exclude", "scale", "model_folders"):
                            info[s2.arg][sub.arg] = _value(s2.value)
                elif sub.arg == "graspable":
                    info["graspable"] = _value(sub.value)
                elif sub.arg == "types":
                    info["types"] = _value(sub.value)
            cats[kw.arg] = info
        return cats
    raise RuntimeError(f"OBJ_CATEGORIES not found in {path}")
