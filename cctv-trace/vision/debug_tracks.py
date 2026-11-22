"""
debug_tracks.py - shows what the pipeline sees for EVERY tracked vehicle, not just the target.

    python -m vision.debug_tracks data\\clips\\clip1.mp4 [out_dir]

Outputs (default out_dir = runs/debug):
  * frame_XXXXX.jpg  annotated frames: vehicle boxes with track id, class and plate-detector confidence
        green  = plate found at the pipeline threshold (conf >= 0.25)
        orange = only a weak plate detection (conf < 0.25): the detector is unsure
        red    = no plate detection at all, even at conf 0.05
  * track_<id>.jpg   the largest crop of each vehicle (check its colour here)
  * a table in the terminal, one row per track
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from vision.pipeline import TRACK_MIN_FPS, VEHICLE_CLASS_MAP, _get_models

PIPELINE_PLATE_CONF = 0.25   # the threshold pipeline.py uses
PROBE_CONF = 0.05            # lower, so weak detections are visible too
SAVE_EVERY = 5               # save every Nth processed frame


def _best_plate_conf(lp_model, frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    res = lp_model.predict(crop, conf=PROBE_CONF, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    return max(float(b.conf[0]) for b in res.boxes)


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else "data/clips/clip1.mp4"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/debug")
    out.mkdir(parents=True, exist_ok=True)

    veh_model, lp_model, _ = _get_models()
    veh_model.predictor = None  # fresh tracker state

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(fps / TRACK_MIN_FPS))

    stats = defaultdict(lambda: {"cls": [], "frames": [], "areas": [], "hits": 0, "best": 0.0, "crop": None, "crop_area": 0.0})
    idx, processed = 0, 0
    while True:
        if not cap.grab():
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                res = veh_model.track(
                    frame, classes=list(VEHICLE_CLASS_MAP), conf=0.35, iou=0.5,
                    persist=True, tracker="bytetrack.yaml", verbose=False,
                )[0]
                vis = frame.copy()
                if res.boxes is not None and len(res.boxes) > 0:
                    for b in res.boxes:
                        if b.id is None:
                            continue
                        tid = int(b.id[0])
                        x1, y1, x2, y2 = map(float, b.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        pconf = _best_plate_conf(lp_model, frame, (x1, y1, x2, y2))

                        s = stats[tid]
                        s["cls"].append(VEHICLE_CLASS_MAP.get(int(b.cls[0]), "?"))
                        s["frames"].append(idx)
                        s["areas"].append(area)
                        if pconf is not None:
                            s["best"] = max(s["best"], pconf)
                            if pconf >= PIPELINE_PLATE_CONF:
                                s["hits"] += 1
                        if area > s["crop_area"]:
                            hh, ww = frame.shape[:2]
                            c = frame[max(0, int(y1)):min(hh, int(y2)), max(0, int(x1)):min(ww, int(x2))]
                            if c.size > 0:
                                s["crop"], s["crop_area"] = c.copy(), area

                        colour = (0, 200, 0) if (pconf or 0) >= PIPELINE_PLATE_CONF else (0, 140, 255) if pconf else (0, 0, 255)
                        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), colour, 4)
                        label = f"#{tid} {VEHICLE_CLASS_MAP.get(int(b.cls[0]), '?')} plate={'-' if pconf is None else f'{pconf:.2f}'}"
                        cv2.putText(vis, label, (int(x1), max(30, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
                if processed % SAVE_EVERY == 0:
                    scale = 1280 / vis.shape[1]
                    small = cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis
                    cv2.imwrite(str(out / f"frame_{idx:05d}.jpg"), small)
                processed += 1
        idx += 1
    cap.release()

    print(f"\n{'track':>5} {'class':<11} {'frames':>6} {'first':>6} {'last':>6} {'med area':>10} {'plate hits':>10} {'best plate conf':>16}")
    for tid, s in sorted(stats.items()):
        cls = max(set(s["cls"]), key=s["cls"].count)
        print(f"{tid:>5} {cls:<11} {len(s['frames']):>6} {s['frames'][0]:>6} {s['frames'][-1]:>6} "
              f"{int(np.median(s['areas'])):>10} {s['hits']:>10} {s['best']:>16.2f}")
        if s["crop"] is not None:
            cv2.imwrite(str(out / f"track_{tid}.jpg"), s["crop"])
    print(f"\nImages saved in {out}")


if __name__ == "__main__":
    main()
