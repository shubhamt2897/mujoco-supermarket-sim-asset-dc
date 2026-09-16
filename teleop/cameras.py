"""List the cameras this machine has, and how fast each one really is.

    python -m teleop.cameras

Frame rate is *measured*, not asked for: CAP_PROP_FPS reports what the driver
intends, which on a webcam in dim light is routinely double what you get. Since
capture is the bottleneck in this pipeline -- 60-85 ms a frame against 38 ms for
the tracking model -- the measured number is the one that decides how the teleop
feels, and it is the number to compare when choosing between a built-in webcam
and a phone.
"""

from __future__ import annotations

import time

import os

# Probing an index that does not exist makes OpenCV shout about DSHOW backends
# and out-of-range indices. Walking the indices IS how you find out how many
# cameras there are, so the noise is expected and unhelpful. This has to be set
# before cv2 is imported -- the messages come from the C++ layer, and
# cv2.setLogLevel() afterwards is too late to stop them.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2          # noqa: E402
import numpy as np  # noqa: E402


def probe(index: int, warmup: int = 6, n: int = 20) -> dict | None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    ok = False
    for _ in range(warmup):
        ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    t0 = time.perf_counter()
    luma = []
    for _ in range(n):
        good, f = cap.read()
        if good:
            luma.append(float(f.mean()))
    ms = (time.perf_counter() - t0) / n * 1000.0
    out = dict(index=index, w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               ms=ms, fps=1000.0 / max(ms, 1e-3),
               luma=float(np.mean(luma)) if luma else 0.0,
               reported=cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return out


def main(max_index: int = 6) -> int:
    print("probing cameras (each takes a second or two) ...\n")
    print(f"{'idx':>3} {'resolution':>12} {'measured':>12} {'reported':>9} {'bright':>7}")
    found = []
    for i in range(max_index):
        r = probe(i)
        if r is None:
            continue
        found.append(r)
        print(f"{r['index']:3d} {r['w']:5d}x{r['h']:<6d} "
              f"{r['fps']:6.1f} fps  {r['ms']:4.0f}ms {r['reported']:8.0f} "
              f"{r['luma']:7.1f}")
    if not found:
        print("no cameras found. Close anything else using the webcam and retry.")
        return 1
    best = max(found, key=lambda r: r["fps"])
    print(f"\nfastest is index {best['index']} at {best['fps']:.0f} fps.")
    print(f"use it with:   python -m teleop --camera {best['index']}")
    print("\nBrightness is the mean pixel value, 0-255. Below about 55 the camera\n"
          "is stretching its exposure to cope, which is what costs the frame rate\n"
          "-- and a long exposure also smears anything moving, which is what makes\n"
          "the landmarks flicker. More light on your face fixes both at once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
