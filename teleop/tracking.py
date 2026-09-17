"""Camera in, `Observation` out, on a thread of its own.

MediaPipe costs 20-50 ms a frame on a CPU. Stepping physics behind that would
drop the sim to tracking rate and make the robot's own dynamics look wrong, so
the tracker runs in a background thread and publishes its latest result into a
slot. The control loop reads whatever is there and never waits. A repeated
observation for one tick is invisible; a sim that stutters is not.

Two backends, both from the mediapipe Tasks API (1.0 dropped the old
`mp.solutions` wrappers entirely):

  holistic   body + both 21-point hands in one pass. The default, and the only
             one that can drive the wrist joints and the grippers, because it
             is the only one that sees fingers. Its hands come pre-labelled as
             the subject's left and right and pre-associated with the body, so
             there is no "which hand is this" step to get wrong when your hands
             cross in front of you.
  pose       body only, using the LITE model, and genuinely faster: 25.6 ms
             against holistic's 38.6 ms, measured. The wrist falls back to the
             four coarse hand landmarks the pose model carries, and the grippers
             hold open. Jitterier than holistic, so not the default.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import TeleopConfig
from .landmarks import Observation

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODELS = {
    "holistic": ("holistic_landmarker.task",
                 "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
                 "holistic_landmarker/float16/latest/holistic_landmarker.task"),
    # Deliberately the LITE pose model, not full.
    #
    # --tracker pose exists to be the fast option, and measured on this machine
    # pose_full costs 40.2 ms against holistic's 38.6 ms -- it is slower than the
    # default while doing strictly less, so choosing it was never rational.
    # pose_lite is 25.6 ms, which is the speed the flag advertises. It is also
    # jitterier, so it is not the default: the wrist and twist joints are
    # already the shakiest part of the mapping.
    "pose": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
}


def ensure_model(kind: str) -> Path:
    """Model bundle on disk, downloading it once if this is the first run."""
    name, url = MODELS[kind]
    path = MODEL_DIR / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} (once) ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    print(f"  -> {path}  ({path.stat().st_size/1e6:.1f} MB)")
    return path


def _measure_frame_ms(cap, n: int = 12) -> float:
    """How long a frame actually takes to arrive, which is not what FPS says."""
    import cv2  # noqa: F401

    for _ in range(4):
        cap.read()
    t0 = time.perf_counter()
    for _ in range(n):
        cap.read()
    return (time.perf_counter() - t0) / n * 1000.0


def _lock_exposure(cap, cfg) -> float | None:
    """Pin the exposure short enough that the camera runs at full frame rate.

    A webcam left on auto-exposure lengthens its frames in dim light, and the
    frame rate falls with it. That is lag no filter downstream can recover,
    because the information never arrived. Measured here: 71 ms a frame on auto
    against 33 ms locked, i.e. half the tracking rate for free.

    Each step is one stop: -6 is 1/64 s, -5 is 1/32 s. Both still allow 30 fps;
    -4 would not, so the walk stops there and hands the camera back to auto
    rather than deliver a picture too dark to track.
    """
    import cv2

    for ev in (-6.0, -5.0):
        # 0.25 is DirectShow's "manual"; 0.75 is "auto".
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, ev)
        for _ in range(6):                   # let the sensor settle
            cap.read()
        luma = []
        for _ in range(3):
            ok, frame = cap.read()
            if ok:
                luma.append(float(frame.mean()))
        if luma and sum(luma) / len(luma) >= cfg.min_luma:
            return ev
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    print("  camera: too dark to lock the exposure, staying on auto.\n"
          "          This costs roughly half the tracking rate, and no filter\n"
          "          downstream can recover it -- the frames never arrive.\n"
          "          Usual cause is a window or lamp BEHIND you: the camera\n"
          "          exposes for the bright background and you go dark. Face\n"
          "          the light instead, and this locks itself at ~30 fps.")
    return None


def _to_array(landmarks, n: int) -> np.ndarray | None:
    """A mediapipe landmark list as an (n, 3) array, or None if it is empty."""
    if not landmarks:
        return None
    out = np.zeros((n, 3))
    for i, lm in enumerate(landmarks[:n]):
        out[i] = (lm.x, lm.y, lm.z)
    return out


def _to_display(landmarks, n: int) -> np.ndarray | None:
    """Image-space landmarks, mirrored left-right to match the displayed picture.

    Only for image coordinates, which are drawn and hit-tested against the air
    buttons on the mirrored display. World coordinates are never mirrored: they
    are the geometry the robot is driven from.
    """
    out = _to_array(landmarks, n)
    if out is not None:
        out[:, 0] = 1.0 - out[:, 0]
    return out


def _visibility(landmarks, n: int) -> np.ndarray | None:
    if not landmarks:
        return None
    v = np.ones(n)
    for i, lm in enumerate(landmarks[:n]):
        got = getattr(lm, "visibility", None)
        if got is not None:
            v[i] = float(got)
    return v


class Tracker:
    """Owns the camera and the landmarker; hands out the most recent frame."""

    def __init__(self, cfg: TeleopConfig | None = None):
        self.cfg = cfg or TeleopConfig()
        self._lock = threading.Lock()
        self._latest: Observation | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.fps = 0.0
        self.error: str | None = None
        self.exposure: float | None = None
        self.frame_ms: float = 0.0
        self._cap = None
        self._landmarker = None
        # The frame most recently handed to mediapipe, so the callback can
        # publish the picture its landmarks belong to along with them.
        self._pending: tuple = (None, time.perf_counter())

    # ---- lifecycle --------------------------------------------------------
    def start(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.cfg.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():                       # DirectShow is Windows-only
            cap = cv2.VideoCapture(self.cfg.camera)
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera {self.cfg.camera}. Close anything else "
                f"using the webcam, or pass --camera 1 for a second one.")
        # MJPG before the size: on most webcams the uncompressed modes are
        # bandwidth limited and cap out well below 30 fps.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.cam_fps)
        # A backlog of stale frames is latency you cannot filter away later.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self.cfg.lock_exposure:
            self.exposure = _lock_exposure(cap, self.cfg)
        self._cap = cap
        self.frame_ms = _measure_frame_ms(cap)
        print(f"  camera: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at "
              f"{1000.0/max(self.frame_ms, 1e-3):.0f} fps "
              f"({self.frame_ms:.0f} ms/frame), exposure "
              f"{'auto' if self.exposure is None else self.exposure}")
        self._landmarker = self._build_landmarker()
        self._thread = threading.Thread(target=self._run, name="tracker", daemon=True)
        self._thread.start()

    def _build_landmarker(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HolisticLandmarker, HolisticLandmarkerOptions,
            PoseLandmarker, PoseLandmarkerOptions, RunningMode)

        path = str(ensure_model(self.cfg.tracker))
        base = BaseOptions(model_asset_path=path)
        if self.cfg.tracker == "holistic":
            return HolisticLandmarker.create_from_options(
                HolisticLandmarkerOptions(
                    base_options=base, running_mode=RunningMode.LIVE_STREAM,
                    min_pose_detection_confidence=0.6,
                    min_pose_landmarks_confidence=0.6,
                    min_hand_landmarks_confidence=0.5,
                    result_callback=self._on_holistic))
        return PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=base, running_mode=RunningMode.LIVE_STREAM,
                num_poses=1, min_pose_detection_confidence=0.6,
                min_tracking_confidence=0.6, result_callback=self._on_pose))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    # ---- the thread -------------------------------------------------------
    def _run(self) -> None:
        import cv2
        import mediapipe as mp

        t0, count = time.perf_counter(), 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            # Track the REAL image; mirror only what is shown.
            #
            # This used to flip the frame before mediapipe saw it, on the belief
            # that landmark labels are anatomical and so unaffected. They are
            # not. A mirror image of you is a different person to the tracker:
            # your left arm appears where a right arm would, so it is labelled
            # "right", and a turn to your left has the depth signature of a turn
            # to the right. The tower yawed the wrong way, and the arms were
            # silently in mirror mode.
            #
            # So mediapipe gets the camera's own frame, which is what it is
            # built for, and the display gets a mirrored copy -- a selfie view,
            # where your raised left hand rises on the left of the screen.
            # Image-space landmarks are mirrored to match in the callbacks.
            shown = cv2.flip(frame, 1)
            self._pending = (shown, time.perf_counter())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._landmarker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(time.perf_counter() * 1000.0))

            count += 1
            dt = time.perf_counter() - t0
            if dt >= 0.5:
                self.fps = count / dt
                t0, count = time.perf_counter(), 0

    # ---- callbacks --------------------------------------------------------
    def _publish(self, obs: Observation) -> None:
        with self._lock:
            self._latest = obs
        self.frames += 1

    def _on_pose(self, result, image, timestamp_ms) -> None:
        frame, t_cap = getattr(self, "_pending", (None, time.perf_counter()))
        pose = result.pose_landmarks[0] if result.pose_landmarks else None
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        self._publish(Observation(
            t=time.perf_counter(),
            pose_image=_to_display(pose, 33), pose_world=_to_array(world, 33),
            pose_vis=_visibility(pose, 33), frame=frame,
            latency_ms=(time.perf_counter() - t_cap) * 1000.0))

    def _on_holistic(self, result, image, timestamp_ms) -> None:
        frame, t_cap = getattr(self, "_pending", (None, time.perf_counter()))
        # HolisticLandmarkerResult carries one subject, so these are plain
        # landmark lists rather than the per-detection lists PoseLandmarker
        # returns -- do not index them.
        pose = result.pose_landmarks
        world = result.pose_world_landmarks
        self._publish(Observation(
            t=time.perf_counter(),
            pose_image=_to_display(pose, 33), pose_world=_to_array(world, 33),
            pose_vis=_visibility(pose, 33),
            hand_world={"left": _to_array(result.left_hand_world_landmarks, 21),
                        "right": _to_array(result.right_hand_world_landmarks, 21)},
            hand_image={"left": _to_display(result.left_hand_landmarks, 21),
                        "right": _to_display(result.right_hand_landmarks, 21)},
            frame=frame,
            latency_ms=(time.perf_counter() - t_cap) * 1000.0))

    # ---- readout ----------------------------------------------------------
    def latest(self) -> Observation | None:
        with self._lock:
            return self._latest


class SyntheticTracker:
    """Same interface, a scripted operator instead of a camera.

    `app.py --source synthetic` runs the entire pipeline and window off this,
    which is how the teleop gets exercised on a machine with no webcam.
    """

    def __init__(self, cfg: TeleopConfig | None = None):
        self.cfg = cfg or TeleopConfig()
        self.fps = 60.0
        self.frames = 0
        self.error = None
        self.exposure = None
        self.frame_ms = 1000.0 / 60.0
        self._t0 = time.perf_counter()
        self.label = ""

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        pass

    def latest(self) -> Observation:
        import cv2

        from . import synthetic as SY

        t = time.perf_counter()
        obs, self.label = SY.scripted(t - self._t0)
        obs.t = t
        self.frames += 1
        frame = np.full((self.cfg.cam_height, self.cfg.cam_width, 3), 30, np.uint8)
        cv2.putText(frame, "SYNTHETIC OPERATOR -- no camera", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 170, 250), 2, cv2.LINE_AA)
        cv2.putText(frame, self.label, (20, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        obs.frame = frame
        return obs


def make_tracker(cfg: TeleopConfig, source: str):
    return SyntheticTracker(cfg) if source == "synthetic" else Tracker(cfg)
