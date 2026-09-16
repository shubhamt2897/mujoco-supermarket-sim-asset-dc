# teleop — drive the tower and both arms with your body

A camera watches you; the lift tower and the OpenArm bimanual arm copy what you
do, in MuJoCo, in real time. One window: you on the left, the robot on the
right, every number in between along the bottom.

---

## Launching it

### 0. One-time setup

The project's own dependencies, plus two more:

```
pip install mediapipe opencv-python
```

Both were checked against the pinned `numpy==2.4.6` and `mujoco==3.13.0` and
move neither. The MediaPipe model bundles (14 MB holistic, 9 MB pose) download
themselves into `teleop/models/` the first time you run it.

Use the environment that already has MuJoCo in it. On this machine that is the
conda env `shelf_sim`:

```
conda activate shelf_sim
cd <repo>
python -m teleop
```

or without activating anything, give the full path to that env's python:

```
C:\Users\<you>\anaconda3\envs\shelf_sim\python.exe -m teleop
```

### 1. Check your camera first

```
python -m teleop.cameras
```

```
idx   resolution     measured  reported  bright
  0   640x480      28.5 fps    35ms       -1   102.5     <- phone as webcam
  1   640x480      16.6 fps    60ms       -1   114.2     <- built-in webcam
```

The **measured** column is the one that matters — `CAP_PROP_FPS` reports what
the driver intends, which in dim light is routinely double what you get. Capture
is the bottleneck in this pipeline, so this number decides how the teleop feels.
It also tells you the index to use, and that index is **not stable**: plugging
in a phone pushed the built-in webcam from 0 to 1.

### 2. Run it

```
python -m teleop                          # default camera, windowed
python -m teleop --camera 0 --fullscreen  # the usual way
```

### 3. Calibrate without touching the keyboard

This matters more than it sounds. Calibration freezes your current pose as the
robot's zero, so a bad one is invisible afterwards — everything tracks
perfectly, just against a crooked reference. And the classic way to get a bad
one is **leaning over to press a key**, which turns one shoulder forward; the
mapping reads torso yaw off your shoulder line, so the lean becomes a permanent
yaw offset for the whole session.

So:

1. Stand back until your head and shoulders are in frame — it says **"got you"**.
2. **Raise a hand into the CALIBRATE box** at the top right of your view and
   hold until the bar fills.
3. **Drop your arms**, face the camera, stand still. A countdown appears, and it
   only ticks while all five checks on the left are green.

Then move: raise an arm, crouch, turn, pinch thumb to index finger.

### 4. Optional, but worth it

Crouch as deep as you actually intend to go and press **`V`** (or its button).
That measures your real range instead of assuming 22% of frame height.

---

## Every option

| flag | what it does |
|---|---|
| `--camera N` | which camera; find N with `python -m teleop.cameras` |
| `--fullscreen` | fill the screen (letterboxed, not stretched) |
| `--scene aisle` | put the shelves and stock back; default is the robot alone |
| `--tracker pose` | body-only model: faster, but no fingers and no grippers |
| `--mirror` | your right arm drives the robot's left |
| `--no-wrist` | hold the three wrist joints at zero |
| `--lock-exposure` | pin a short exposure: ~2x the frame rate, fixed at today's light |
| `--viewer` | also open the interactive MuJoCo window |
| `--record PATH` | log every tick to `PATH.npz` plus frames, to review afterwards |
| `--source synthetic` | a scripted operator; no camera needed at all |
| `--render N` / `--panel N` | size of the robot half / of both halves |
| `--pretty` | shadows and anti-aliasing (slower) |
| `--headless N --save P` | run N ticks with no window and save one frame |

## Every control

The buttons along the bottom do the same things as the keys.

| key | |
|---|---|
| `C` | calibrate now (refuses a bad pose, and starts the countdown instead) |
| `T` | hands-free countdown — same as the CALIBRATE air target |
| `V` | set your crouch depth, at the bottom of a squat |
| `SPACE` | engage / disengage. This is the deadman |
| `M` | mirror the mapping |
| `W` | wrist tracking on/off |
| `R` | reset the robot to home |
| `H` | hide the air target |
| `F` | camera follows the carriage (off by default, so the lift stays visible) |
| `[` `]` | swing the robot view round |
| `,` `.` | tilt it |
| `-` `+` | zoom |
| `F11` | full screen (button only, on some builds) |
| `Q` / `Esc` | quit |

## If something is wrong

| symptom | cause |
|---|---|
| nothing moves | not calibrated, or disengaged — check the pill at bottom left |
| robot sits at a yaw offset | calibrated while leaning; recalibrate hands-free |
| laggy | the camera. Run `teleop.cameras` and see the timing table below |
| landmarks flicker | long exposure blurs motion. More light, or a better camera |
| "hips assumed" in the HUD | hips not visible; fine, but leaning is not tracked |
| carriage bottoms out too easily | crouch fully and press `V` |

---

## What drives what

| you | the robot |
|---|---|
| stand up / crouch down | the carriage rides its 0.70 m of travel, top to bottom |
| turn your shoulders | the tower slews the same way |
| your shoulder, elbow, wrist | that arm's 7 joints copy the angles |
| pinch thumb to index finger | that side's gripper closes |

Your left drives the robot's left. You are looking at a mirrored picture of
yourself, so raising your left hand raises the hand on the left of the screen —
as if you were inside the robot rather than facing it. `M` swaps it if you would
rather work against a reflection.

## How it works

```
camera ──► mediapipe ──► Observation ──► Retargeter ──► Targets ──► MuJoCo
        (its own thread)   landmarks      joint angles   ctrl[]    position
                                                                  actuators
```

Tracking runs on its own thread. MediaPipe costs 20–50 ms a frame on a CPU, and
stepping physics behind that would drag the simulation down to tracking rate and
make the robot's own dynamics look wrong. The control loop reads the most recent
result and never waits.

### Angle retargeting, not IK

The obvious approach — track the wrist, run IK on the end effector — is the
wrong one here. The arm is 7-DOF with a tightly limited wrist (joint 6 is ±45°,
joint 4 cannot go below 0), so a position-only solve has a null space it wanders
through and no way to say *my elbow is out to the side*. It also cannot be
checked offline.

The arm is built anatomically — shoulder pitch, shoulder abduction, humeral
twist, elbow, forearm twist, wrist pitch, wrist roll — so your joint angles map
onto it directly. They come out in **closed form** from three landmarks per arm:
no iteration, no null space, and it round-trips against MuJoCo's own forward
kinematics to about 1e-13 (see `selftest.py`).

The two frames are deliberately identical — *x forward, y to the left, z up* —
for the robot's arm base and for the torso frame built from your shoulders and
hips. So a direction in your torso frame **is** a direction in the arm's base
frame, with nothing to convert and nothing to get backwards.

### Sitting at a desk

Your hips are used when they are visible — the shoulder-to-hip line is the best
conditioned estimate of which way is up. But at a desk they are cropped or
behind it, and MediaPipe reports them at ~0.3 visibility with positions
extrapolated off the bottom of the frame, which tilts the whole torso frame.

So hips are *optional*: when they are not confidently seen, the camera's own up
axis stands in and the HUD reads **hips assumed**. The cost is that leaning is
no longer tracked; the benefit is that the teleop works from a chair, which is
where it usually gets tested. Requiring them meant it simply never engaged.

### Why the lift reads image space and the yaw reads world space

MediaPipe gives two landmark sets. The world set is metric but **hip-centred**,
so it travels with you and cannot tell standing from crouching. The image set is
anchored to the picture and can: crouch, and your shoulders move down the frame.
So lift is measured there, and everything angular is measured in the world set,
where it is scale- and position-invariant.

### Filtering

Every output goes through a [One Euro filter](https://gery.casiez.net/1euro/) —
it smooths hard when you are still and barely at all when you move, which is
what stops the hands jittering at rest without adding lag when they don't. Each
channel is then rate-limited, so a tracking dropout becomes a slew rather than a
full-torque lunge on a position actuator.

Turn the knobs in `config.py`: raise `beta_*` if it feels sluggish, lower
`mincutoff_*` if it twitches at rest.

### Commands go to actuators, never to qpos

`SimBridge.apply` only ever writes `data.ctrl`. The arms are driven by the same
position actuators a real controller would use, so you see the real tracking
lag, the real sag under gravity and the real contact response. Writing `qpos`
would look perfect and prove nothing.

---

## Testing it without a camera

`selftest.py` checks 53 things in four layers, each depending on the last:

1. **Frame conventions**, re-measured off the compiled model every run — so if
   the arm is ever swapped, this fails loudly instead of silently mis-aiming.
2. **Forward kinematics** — the 30-line numpy chain against MuJoCo's own, over
   random configurations. Agreement here means the axis signs are right.
3. **The inverse** — random angles → FK → solve → the same angles back. This is
   the one that catches a flipped sign, which is otherwise invisible: a wrong
   sign still produces smooth, plausible motion, just mirrored.
4. **The mapping end to end**, on a fabricated operator, then actually driving
   the simulated robot with it.

`--source synthetic` runs a scripted operator — stand, reach, crouch, turn,
pinch — through the real window, so the UI and the sim can be exercised on a
machine with no webcam.

```
python -m teleop --source synthetic
python -m teleop --source synthetic --headless 300   # saves a frame, no window
```

---

## Files

| | |
|---|---|
| `retarget.py` | the maths: landmarks → joint angles. Pure numpy, no mediapipe, no mujoco |
| `tracking.py` | camera + mediapipe on a background thread |
| `simbridge.py` | builds the robot, applies targets, renders. All the MuJoCo |
| `overlay.py` | the window: skeleton, robot view, HUD |
| `synthetic.py` | a fabricated operator, for testing with no camera |
| `selftest.py` | the checks above |
| `config.py` | every tunable |
| `models/` | mediapipe model bundles, downloaded on first run |

The split is deliberate: `retarget.py` imports neither mediapipe nor mujoco, so
the maths can be checked against the simulator without the tracker, and the
tracker can be run without the maths.

## Requirements

Beyond the project's own dependencies:

```
pip install mediapipe opencv-python
```

Both were verified to install alongside the pinned `numpy==2.4.6` and
`mujoco==3.13.0` without moving either. The model bundles (~14 MB holistic,
~9 MB pose) download to `models/` on first run.

`--tracker pose` is faster but body-only: no fingers, so the grippers hold open
and the wrist falls back to the four coarse hand landmarks the pose model
carries.

## If it feels laggy, it is almost certainly your lighting

Measured on this machine, per frame:

| stage | cost |
|---|---|
| `cap.read()`, auto-exposure, backlit | **60–85 ms** |
| `cap.read()`, exposure locked | **33 ms** |
| holistic inference | 38 ms |
| pose inference | 23 ms |
| robot render | 26 ms |
| physics | 1 ms |

The camera dominates everything, and it is the one nobody looks at. In dim
light a webcam lengthens its exposure to brighten the picture, and a longer
exposure *is* a longer frame — the frame rate halves before a single landmark
has been computed. That is lag no amount of filtering downstream can recover,
because the information never arrived.

`--lock-exposure` pins a short exposure and roughly doubles the tracking rate.
It is **opt-in, not the default**: locking picks one exposure at startup, and
room light does not hold still — a value that works in the morning is wrong by
the afternoon, and a rig you have to relaunch when the sun moves is worse than
a slow one. Auto adapts; you pay for it in frames.

If you do lock it and the picture is too dark, it loosens a stop and then gives
up and hands the camera back to auto, because a picture too dark to track is
worse than both.

**The usual cause of it being too dark is a window or lamp behind you.** The
camera exposes for the bright background and you become a silhouette; locking
then drops pose detection from 100% to **0%**, and brightening in software does
not rescue it — measured, luma 27 → 65 with no recovery at all. Turning to face
the light is what actually fixes it.

Note that `--tracker pose` will *not* help here: inference is not the
bottleneck, capture is, so the faster model buys nothing until the camera is fixed.

## Why it still shakes, and what was done about it

Two recorded sessions, same operator, same movements, different cameras:

| | laptop webcam | phone as webcam |
|---|---|---|
| tracking latency (median) | 40.4 ms | **17.9 ms** |
| tracking latency (p90) | 67.9 ms | **33.5 ms** |
| torso landmark jitter (median) | 0.0112 | **0.0053** |

So the camera was more than half of it: **landmark jitter halved** and latency
dropped 2.3x just by using a phone. A long exposure smears anything moving, and
a smeared frame puts the landmarks somewhere slightly different every time.

The rest is in the retargeting, and the session log says exactly where. Breaking
the commands into genuine motion and high-frequency residual, per joint:

```
joint   motion   shake   ratio
  j1    18.6      3.9    0.21     shoulder pitch
  j2    11.7      2.1    0.18     shoulder abduction
  j3    22.1      4.0    0.18     humeral twist
  j4    11.2      3.3    0.29     elbow
  j5    32.2      9.6    0.30     forearm twist   <- worst
  j6    23.2      6.5    0.28     wrist pitch
  j7    13.1      5.4    0.41     wrist roll      <- worst ratio
```

Two things that rules out, both worth checking before believing them:

- **It is not the robot.** Commanded shake 4.97°, measured shake 4.77°, tracking
  error 1.35°. The actuators faithfully reproduce a shaky command; they are not
  adding oscillation of their own.
- **It is not the elbow singularity.** `q3` is ill-conditioned when the elbow is
  straight, which was the obvious suspect — but the elbow sat at a median of 88°
  and was below 20° for 0–1% of ticks. Wrong hypothesis, discarded.

It is **the wrist chain**, and the reason is geometric: the palm axis is read
across the ~4 cm between two knuckle landmarks, so a millimetre of landmark
noise is degrees of roll. The twists (`j3`, `j5`) are the least observable
degrees of freedom on a limb — rotation *about* an axis you are estimating *from*
that axis.

The fix was to filter **the direction vectors going in**, not the joint angles
coming out. Smoothing an angle means smoothing the output of a nonlinear map
that has already amplified the noise; smoothing the unit vectors first is far
better conditioned. The hand directions get their own, much heavier setting.

Measured on synthetic landmarks with a controlled 8 mm of noise:

| joint | shake before | after | |
|---|---|---|---|
| j1 | 2.69° | 1.74° | −35% |
| j3 | 5.26° | 3.90° | −26% |
| j5 | 5.63° | 4.10° | −27% |
| j6 | 1.93° | 0.48° | **−75%** |
| j7 | 1.94° | 0.46° | **−76%** |
| mean | 2.81° | **1.74°** | **−38%** |

with genuine motion fully retained — this is noise removal, not damping.

`j3` and `j5` remain the residual. They are the limb twists, and no amount of
filtering makes an unobservable degree of freedom observable; `--no-wrist` pins
the wrist joints at zero if you would rather not see them move at all.

## Which way should the robot face?

The default view sits **behind the robot, looking the same way it looks** —
MuJoCo azimuth 90, since the robot faces world +y. That is not a cosmetic
choice; it is the only view that agrees with the mapping:

|  | over-the-shoulder (az 90, default) | face to face (az 270) |
|---|---|---|
| robot's left arm appears | on **your left** | on your right |
| a forward reach | goes **away into the screen**, like yours | comes toward you |
| grippers on a forward reach | partly hidden behind the torso | fully visible |

The mapping is **embodiment**: your left drives the robot's left, your forward
is its forward, as though you were inside it. A face-to-face view silently
contradicts that — the correspondence is still correct, but everything you see
is mirrored, and you spend the session translating. Over-the-shoulder costs you
some sight of the grippers, which `[` and `]` fix on demand.

`M` switches the *mapping* to a mirror if you would rather work against a
reflection. That is a different thing from the camera angle, and the two are
independent.

## Calibrating without touching anything

Calibration freezes whatever you are doing as the robot's zero, which makes a
bad calibration pose invisible afterwards — everything still tracks perfectly,
just against a crooked reference.

The way it actually goes wrong is **reaching for the keyboard**. Leaning toward
the machine turns one shoulder forward, and a rotated shoulder line *is* torso
yaw as far as the mapping is concerned, so the zero gets taken mid-twist. A
mouse click has exactly the same problem — you lean the same way to reach the
mouse.

So there is a **CALIBRATE target on the camera image**: raise a hand into it and
hold, and a bar fills as you hold. It does not calibrate on contact, because
that would capture you with an arm up. It starts a two-phase countdown:

1. **ungated** (`ready_s`, 2.5 s) — "put your arms down". The clock runs
   regardless; you have just had a hand in the air and need time to lower it.
2. **gated** — the clock only ticks while five checks are green: in frame,
   square to camera, shoulders level, arms down, holding still. It pauses and
   says which one failed.

A timeout (18 s) calibrates anyway rather than blocking forever, and says what
was off. Once calibrated the target shrinks to a small corner box with a longer
dwell, so it stays reachable without being somewhere you will hit by accident;
`H` hides it entirely.

The gate thresholds are **measured, not guessed**. The first set were guesses
and blocked calibration completely — in a recorded 79-second session `still`
passed 6% of ticks and the robot was driven for **0% of the session**. Watching
the wrists was the mistake: hands jitter several times more than shoulders, and
calibration only reads the torso. After moving stillness to the torso and
relaxing `square` from 22° to 30°, the next session drove 92% of its ticks.

## Choosing a camera

```
python -m teleop.cameras
```

Lists every camera with its **measured** frame rate, because `CAP_PROP_FPS`
reports what the driver intends, which in dim light is routinely double what you
get. Capture is the bottleneck in this pipeline, so that measured number is what
decides how the teleop feels.

A phone used as a webcam (Iriun, DroidCam, Camo) is usually a large upgrade over
a built-in laptop camera: a bigger sensor needs less exposure, which means both
a higher frame rate and less motion blur. Find its index with the command above
and pass `--camera N`.

## Keyboard or on-screen buttons?

Both, because neither is sufficient alone:

- **Standing back far enough to be in frame, you can reach neither.** That is
  what `T` / "5s hands-free" is for — it counts down on screen in letters big
  enough to read across a room, then calibrates.
- **Up close, buttons are discoverable** in a way an unlabelled keymap is not,
  and they show state: engage, mirror, wrist and follow light up when active.
- **Once you know it, the keyboard is faster**, and it is the only sensible way
  to drive the camera nudges, which you hold down and repeat.

The buttons and the keys dispatch through one `act()` function, so they cannot
drift apart.
