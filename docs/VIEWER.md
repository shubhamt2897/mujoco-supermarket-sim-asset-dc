# Running the supermarket scene in the MuJoCo viewer

How to open the shelf-restocking scene, what the pieces are, and what to do when
it misbehaves.

---

## Quick start

```bash
cd "D:/New folder/da_supermarkt"

python view.py                    # scene only, no side panels, fast rendering
python view.py --robot            # with the bimanual arm attached
python bench.py                   # tower + arms alone, control sliders on
```

Close the window to exit. The command blocks until you do.

If plain `python` is not the conda environment, call it by full path:

```bash
"C:/Users/shubh/anaconda3/envs/shelf_sim/python.exe" view.py --robot
```

### Prerequisites

Conda environment `shelf_sim`: Python 3.11, `mujoco` 3.13.0, `numpy`, `imageio`,
`openarm_mujoco` 2.2.0. CPU only - no torch, jax or CUDA anywhere in this project.

The scene file `out/scene.xml` must exist. If it does not, generate it:

```bash
python scene.py
```

---

## `view.py` flags

| flag | effect |
|---|---|
| *(none)* | scene only, panels hidden, fast rendering |
| `--robot` | attach the OpenArm v2 bimanual arm at the `arm_mount` site |
| `--ui` | show the left and right control panels |
| `--pretty` | keep full shadows and anti-aliasing (much slower) |
| `--passive` | you drive the physics loop instead of MuJoCo |
| `--scene PATH` | load a different MJCF (default `out/scene.xml`) |

Flags combine: `python view.py --robot --passive --ui`.

On startup it prints what it loaded, which is the fastest way to confirm you got
what you expected:

```
scene    : D:\New folder\da_supermarkt\out\scene.xml
robot    : attached
bodies   : 390   geoms: 820   dof: 192
rendering: fast (no shadows/AA)
panels   : off -- press Tab to toggle
```

---

## `bench.py` - the tower on its own

The full aisle is a poor place to test joints: 392 bodies, slow to render, and
the shelves and roll cage physically obstruct the slew. `bench.py` brings up the
tower and arms on a bare floor with nothing else in the world.

```bash
cd "D:/New folder/da_supermarkt"

python bench.py                   # tower + arms, control panel on, fast
python bench.py --sweep           # scripted joint sweep, prints numbers, no window
python bench.py --no-ui           # hide the panels
python bench.py --pretty          # full shadows and anti-aliasing
```

24 bodies and 101 geoms against 392/824 for the aisle, so it renders instantly
and the sliders respond immediately.

### Driving the joints

The control panel is **on by default here** - that is where the actuator sliders
are, in the right panel under **Control**, and dragging them is the point.
**Ctrl+drag** a slider to move a joint.

| # | actuator | range |
|---|---|---|
| 0 | `tower_yaw_ctrl` | -3.1416 … +3.1416 rad = -180…+180° |
| 1 | `tower_lift_ctrl` | 0.000 … 0.700 m |
| 2-17 | arm joints and grippers | per joint |

It prints this table with degree equivalents on startup, so there is no need to
convert in your head.

### `--sweep`

Drives both base joints through their range with no window and reports tracking
error and sag. Use it after any edit to `tower.xml`:

```
yaw  : commanded +/-180 deg, max error 0.0002 deg
lift : 0.00 m sag 0.31 mm, 0.175-0.700 m sag 1.74 mm
both : (+90 deg, 0.70 m) and (-90 deg, 0.00 m) reached exactly
```

The bench goes through the same `robot.assemble()` path as the full scene and
uses the same `tower_base_site` name, so what you drive here is the model that
ends up in the aisle - not a separate mock. `out/bench.xml` is a generated bare
world (floor, lights, three cameras, the mount site) and is rewritten each run;
edit `BENCH_XML` in `bench.py`, not the generated file.

---

## How a MuJoCo sim is launched

Three objects, always the same three.

### 1. `MjModel` - the compiled scene

Everything that does not change: geometry, masses, actuators, cameras, lights.
Built once from the MJCF file.

```python
model = mujoco.MjModel.from_xml_path("out/scene.xml")
```

### 2. `MjData` - the state

Everything that does change: `qpos` (positions), `qvel` (velocities), `ctrl`
(actuator targets), `contact`. One `MjModel` can back many `MjData` - that is how
you run parallel rollouts later.

```python
data = mujoco.MjData(model)
```

### 3. A viewer - a window drawing `data` against `model`

Two kinds, and the difference decides what you can build on top.

**Managed.** MuJoCo runs the physics loop internally and blocks until the window
closes. Good for looking around, useless for control.

```python
mujoco.viewer.launch(model, data)
```

**Passive.** Returns immediately and hands back a handle. *You* call `mj_step`,
then `sync()` to push the new state to the window.

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)   # advance physics by one timestep
        viewer.sync()                 # show the new state
```

`mj_step` is the entire simulation: it reads `data.ctrl`, solves contacts,
integrates, and writes back into `data`.

**Use passive for anything real.** Inverse kinematics, a policy, a data recorder
and scripted motion all go *inside* that `while` loop, between `mj_step` and
`sync`. The managed viewer gives you nowhere to put them. `view.py --passive`
exists so there is a working loop to copy.

### The zero-code option

Any MJCF file can be opened without writing Python:

```bash
python -m mujoco.viewer --mjcf="out/scene.xml"
```

It always shows the panels, and it **cannot** load the robot - the arms are
joined to the scene in memory and never written to disk, so there is no file for
this command to open.

---

## Hiding the control panels

```python
mujoco.viewer.launch(model, data, show_left_ui=False, show_right_ui=False)
```

Both `launch` and `launch_passive` accept these. At runtime, **Tab** toggles the
left panel and **Shift+Tab** the right, so hiding them costs you nothing.

---

## Rendering performance

The scene is authored for offscreen dataset renders, which makes it heavy for
interactive viewing. `view.py` downgrades four display-only fields *after*
compilation, so `out/scene.xml` keeps full quality and only the window is
affected. `--pretty` opts back in.

```python
model.vis.quality.shadowsize = 0   # shadow map - the big one
model.vis.quality.offsamples = 0   # anti-aliasing
model.light_castshadow[:] = 0      # no light casts a shadow
model.mat_reflectance[:] = 0       # no mirror-like floor
```

Measured at 1280×720 on the `over_shoulder` camera:

| setting | ms/frame | fps |
|---|---|---|
| as generated | 3469 | 0.3 |
| shadows off only | 931 | 1.1 |
| anti-aliasing off only | 1630 | 0.6 |
| **all four off** | **363** | **2.8** |

A 9.6× improvement. Shadows dominate; `scene.py` sets `shadowsize="4096"` and
`offsamples="8"`, which are right for stills and punishing for a live window.

Re-measured after the over-detailed product meshes were simplified (below), same
camera and settings: **181 ms, 5.5 fps**.

### The deeper cost - and what was done about it

Display flags only go so far. Two scanned products shipped with absurd detail
for objects about 7 cm across: `canned_food_9` at 197,956 triangles and
`yogurt_3` at 194,470. With their copies on the shelves they were **79% of the
7.49 million triangles drawn every frame**.

They were simplified rather than dropped from stocking, with
[`tools/simplify_meshes.py`](../tools/simplify_meshes.py). Seven visual meshes
over 12,000 triangles were reduced; smaller ones were deliberately left alone.

| | before | after |
|---|---|---|
| triangles drawn per frame | 7,490,650 | 1,650,062 |
| `canned_food_9` | 197,956 | 4,000 |
| `yogurt_3` | 194,470 | 8,000 |
| `assets/products/` on disk | 214.9 MB | 135.0 MB |
| `over_shoulder`, 1280×720, display options off | 363 ms | 181 ms |

Only visual meshes changed. Physics is unaffected: products collide as fitted
boxes and cylinders, never as meshes, and each product's density was rescaled so
its mass still matches its target exactly (`tools/verify_products.py` passes).

Each simplified mesh was rendered from four sides against its original and also
**checked by eye** - the automatic comparison alone is not enough. It passed a
one-pixel seam line on `canned_food_18` and a warped label on `boxed_food_4`,
both of which were reverted.

### MuJoCo does not cull

Every geom is sent to the GPU whatever the camera is pointed at. `tower_eye`,
which sees one bay, cost the same as `overhead`, which sees the whole aisle. The
teleop window culls geoms outside each camera's view before rendering
(`teleop/simbridge.py`), which took `tower_eye` from 141 ms to **17 ms** with a
pixel-identical image. The managed viewer and `live.py` do not do this.

### Physics speed (separate from rendering)

Measured with `tools/check_scene.py` and the robot attached: **~5.2× realtime**
(about 2,580 steps/s at a 2 ms timestep), 119 contacts, no NaNs. Physics is not
the bottleneck for viewing - rendering is.

---

## Regenerating the scene

`out/scene.xml` is generated, not hand-edited. Change the dataclasses at the top
of `scene.py` (`ShelfDims`, `StockingDims`, `GapSpec`, `RollCageDims`,
`RobotMountDims`, `FloorDims`) and rebuild:

```bash
python scene.py                   # writes out/scene.xml
python tools/check_scene.py       # contacts, NaN check, drift, steps/s
python tools/render_cams.py       # one PNG per camera into out/shots/
python robot.py --shots           # attach arms, settle 3 s, report, save shots
```

`robot.py` reads `out/scene.xml`, so **regenerate the scene before running it**
after any change.

---

## Cameras

The scene has four fixed cameras: `aisle`, `shelf_front`, `overhead`,
`over_shoulder`. With `--robot` three more arrive with the models:

| camera | comes from | what it sees |
|---|---|---|
| `tower_ledge` | `tower.xml` | head view from the shoulder ledge; yaws and lifts with the arms |
| `camera_wrist_left` | the arm | left wrist |
| `camera_wrist_right` | the arm | right wrist |

`bench.py` instead provides `bench_front`, `bench_side` and `bench_top`, plus
`tower_ledge` and the two wrist cameras.

In the viewer, cycle cameras with the **`[`** and **`]`** keys.

### Depth (RGB-D)

Any camera can be rendered as depth as well as colour:

```python
from robot import build, render_rgbd
bot = build()
info = render_rgbd(bot.model, bot.data, "tower_ledge", Path("out/shots"))
```

It writes three files: `<cam>_rgb.png`, `<cam>_depth.npy` (**metres**, float32 -
this is the one a policy or point-cloud step wants) and `<cam>_depth.png`
(normalised, for looking at). The returned dict carries min/median/max depth and
the fraction of pixels closer than `far_clip`, which is the quick way to spot a
camera buried inside geometry.

---

## Troubleshooting

**`python` is not the environment.** Use the full interpreter path
(`C:/Users/shubh/anaconda3/envs/shelf_sim/python.exe`) or `conda activate shelf_sim`
first.

**`Error opening file '../assets/products/...'`.** `out/scene.xml` refers to
meshes by a path relative to itself (`meshdir="../assets/products"`). It only
resolves from inside `out/`. Copying the XML elsewhere breaks every mesh - write
generated scenes into `out/`, or regenerate at the new location.

**`conda run` fails on a `-c` snippet.** `conda run` rejects arguments containing
newlines. Call the environment's `python.exe` directly instead, and keep any
`-c` snippet on one line.

**`UnboundLocalError: cannot access local variable 'mujoco'`.** Somewhere a
function does `import mujoco.viewer` locally, which rebinds the name `mujoco` for
that whole function. Use `import mujoco.viewer as mj_viewer` instead.

**`Image width 1280 > framebuffer width 640`.** Offscreen rendering is capped by
the model's declared buffer. `scene.py` emits
`<visual><global offwidth="1920" offheight="1080"/></visual>`; a scene without it
will not render above 640×480.

**Window opens black or empty.** The camera is probably inside geometry. Press
**Esc** to return to the free camera, then `[` / `]` to step through the fixed
cameras.

**Viewer seems frozen.** It is not - `launch` blocks the terminal until the
window closes. That is expected. Run it in the background if you need the shell
back.

---

## File map

| file | role |
|---|---|
| `view.py` | opens the viewer - panels, render quality, managed vs passive |
| `bench.py` | tower + arms alone, for testing the joints |
| `scene.py` | generates `out/scene.xml` from dataclasses |
| `tower.xml` | **hand-authored** lift tower: rotary joint, prismatic lift, shoulder bracket, `arm_mount`, `tower_ledge` camera. `scene.py` never generates or overwrites it |
| `robot.py` | joins scene + tower + arm in memory; `Robot` handle with `set_base`/`get_base`; `render_rgbd` |
| `out/bench.xml` | generated bare world for the bench - rewritten each run |
| `tools/check_scene.py` | contacts, NaN check, drift, steps/s |
| `tools/render_cams.py` | one PNG per camera |
| `tools/product_library.py` | reads product MJCFs, picks collision primitives |
| `tools/product_spec.py` | the eight categories, corrected masses, density bands |
| `out/scene.xml` | generated - do not hand-edit |
| `assets/products/` | extracted RoboCasa meshes - edit only through `tools/simplify_meshes.py`, and look at its `--sheet` before committing |
