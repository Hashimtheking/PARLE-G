"""
pipeline.py - Person A: vision pipeline.

run_vision(video_path, settings, run_dir) -> VisionOutput   (see schema.py)

Changes vs. the first version
  1. Tracker state is reset at the start of every run (no leaking between uploads).
  2. Two passes: pass 1 tracks and stores BOXES only; pass 2 re-reads just the target
     vehicle's frames. No full frames held in RAM; skipped frames use grab() (fast).
  3. Tracking runs at >= TRACK_MIN_FPS so ByteTrack sees small motion between frames.
  4. Plate detection on every target frame, then OCR only on the MAX_OCR_FRAMES sharpest.
  5. Plate boxes are padded; sharpness is measured at a fixed height (resolution independent).
  6. OCR words are grouped into lines before ordering left-to-right.
  7. quality_weight = sharpness / sum(sharpness of used frames) (no double-counting OCR conf).
  8. Stronger enhancement (upscale to >=128 px height, CLAHE, mild unsharp mask).
  9. If no frame passes the sharpness gate, the sharpest few are used and a note says so.
 10. vision.json is written into run_dir.

NOTE for Person B: OCR character confidence is still WORD-LEVEL (every char in a word shares
the word's confidence), and plate strings vary in length between frames. Align before voting.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from schema import Box, CharReading, FrameReading, Settings, VisionOutput

# COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

TRACK_MIN_FPS = 10.0         # tracking rate floor (ByteTrack needs small motion between frames)
MAX_OCR_FRAMES = 12          # OCR only the N sharpest plate frames (CPU time)
FALLBACK_FRAMES = 3          # used when no frame passes the sharpness gate
PLATE_PAD_FRAC = 0.0        # padding around the detected plate box
SHARPNESS_NORM_HEIGHT = 64   # plate crops are resized to this height before measuring sharpness
ENH_MIN_HEIGHT = 128         # plate crops are upscaled to at least this height for OCR
LEFT_TRIM_FRAC = 0.0         # set to ~0.08 to cut the blue "IND" strip; test on your clips first
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Lazy global model cache so repeated runs (Streamlit) do not reload weights
_MODELS_CACHE: dict = {}


def _get_models(models_dir: str = "models", use_gpu: bool = False):
    if not _MODELS_CACHE:
        m_dir = Path(models_dir)
        _MODELS_CACHE["veh"] = YOLO(str(m_dir / "yolov8s.pt"))
        _MODELS_CACHE["lp"] = YOLO(str(m_dir / "lp_detector.pt"))
        _MODELS_CACHE["ocr"] = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            model_storage_directory=str(m_dir / "easyocr"),
            download_enabled=False,
        )
    return _MODELS_CACHE["veh"], _MODELS_CACHE["lp"], _MODELS_CACHE["ocr"]


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def _sharpness(plate_bgr: np.ndarray) -> float:
    """Laplacian variance measured at a fixed plate height so it is comparable across clips."""
    if plate_bgr is None or plate_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 2 or w < 2:
        return 0.0
    new_w = max(2, int(round(w * SHARPNESS_NORM_HEIGHT / h)))
    gray = cv2.resize(gray, (new_w, SHARPNESS_NORM_HEIGHT), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _enhance_plate(plate_bgr: np.ndarray) -> np.ndarray:
    """Grayscale -> optional left trim -> upscale -> CLAHE -> mild unsharp mask."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    if LEFT_TRIM_FRAC > 0:
        gray = gray[:, int(gray.shape[1] * LEFT_TRIM_FRAC):]
    h = gray.shape[0]
    if h < ENH_MIN_HEIGHT:
        scale = ENH_MIN_HEIGHT / max(h, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


# ----------------------------------------------------------------------------
# OCR helpers
# ----------------------------------------------------------------------------
def _order_results(results: list) -> list:
    """Group OCR boxes into text lines (by vertical position), then sort each line left-to-right."""
    if not results:
        return results
    heights = [abs(r[0][2][1] - r[0][0][1]) for r in results]
    tol = 0.5 * float(np.median(heights))
    results = sorted(results, key=lambda r: r[0][0][1])
    lines: List[list] = []
    cur = [results[0]]
    for r in results[1:]:
        if abs(r[0][0][1] - cur[0][0][0][1]) <= tol:
            cur.append(r)
        else:
            lines.append(cur)
            cur = [r]
    lines.append(cur)
    return [r for line in lines for r in sorted(line, key=lambda r: r[0][0][0])]


def _read_plate_chars(reader: easyocr.Reader, enhanced_gray: np.ndarray) -> Tuple[str, List[CharReading]]:
    """
    EasyOCR returns ONE confidence per word; it is copied to every character of that word.
    len(raw_text) == len(chars) as schema.py requires. Nothing is corrected or guessed here.
    """
    results = reader.readtext(enhanced_gray, allowlist=OCR_ALLOWLIST, detail=1)
    if not results:
        return "", []
    raw_chars: List[str] = []
    readings: List[CharReading] = []
    for _bbox, text, conf in _order_results(list(results)):
        cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
        c = round(min(max(float(conf), 0.0), 1.0), 4)
        for ch in cleaned:
            raw_chars.append(ch)
            readings.append(CharReading(char=ch, conf=c))
    return "".join(raw_chars), readings


# ----------------------------------------------------------------------------
# Plate detection on one frame
# ----------------------------------------------------------------------------
def _detect_plate(lp_model, frame: np.ndarray, vbox: Tuple[float, float, float, float]):
    """Returns (plate_crop, plate_box_in_frame_pixels, skip_reason)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = vbox
    px, py = (x2 - x1) * 0.05, (y2 - y1) * 0.05
    vx1, vy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    vx2, vy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    veh = frame[vy1:vy2, vx1:vx2]
    if veh.size == 0:
        return None, None, "empty vehicle crop"

    res = lp_model.predict(veh, conf=0.25, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, None, "no plate detected"

    best = max(res.boxes, key=lambda b: float(b.conf[0]))
    lx1, ly1, lx2, ly2 = map(float, best.xyxy[0])
    padx, pady = (lx2 - lx1) * PLATE_PAD_FRAC, (ly2 - ly1) * PLATE_PAD_FRAC
    lx1, ly1 = max(0.0, lx1 - padx), max(0.0, ly1 - pady)
    lx2, ly2 = min(float(veh.shape[1]), lx2 + padx), min(float(veh.shape[0]), ly2 + pady)

    crop = veh[int(ly1):int(ly2), int(lx1):int(lx2)]
    if crop.size == 0:
        return None, None, "empty plate crop"
    box: Box = (vx1 + lx1, vy1 + ly1, vx1 + lx2, vy1 + ly2)
    return crop, box, None


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def run_vision(
    video_path: str,
    settings: Settings,
    run_dir: str,
    models_dir: str = "models",
    use_gpu: bool = False,
) -> VisionOutput:
    veh_model, lp_model, ocr_reader = _get_models(models_dir=models_dir, use_gpu=use_gpu)

    # Reset the tracker so IDs/state from a previous clip never leak into this one.
    veh_model.predictor = None

    video_file = Path(video_path)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = round(total_frames / fps, 3) if total_frames > 0 else 0.0

    track_fps = max(settings.sample_fps, TRACK_MIN_FPS)
    stride = max(1, round(fps / track_fps))
    max_frame_idx = int(settings.max_clip_seconds * fps)

    run_path = Path(run_dir)
    frames_dir = run_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    notes: List[str] = []
    if total_frames > max_frame_idx:
        notes.append(f"Clip capped at {settings.max_clip_seconds}s ({max_frame_idx} frames).")

    # ---------------- Pass 1: track vehicles, store boxes only ----------------
    tracks: Dict[int, dict] = {}
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx > max_frame_idx:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                res = veh_model.track(
                    frame,
                    classes=list(VEHICLE_CLASS_MAP.keys()),
                    conf=0.35,
                    iou=0.5,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        if box.id is None:
                            continue
                        tid = int(box.id[0])
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        t = tracks.setdefault(tid, {"track_id": tid, "classes": [], "observations": []})
                        t["classes"].append(VEHICLE_CLASS_MAP.get(cls_id, "car"))
                        t["observations"].append({
                            "frame_idx": idx,
                            "timestamp_s": round(idx / fps, 3),
                            "vehicle_box": (x1, y1, x2, y2),
                            "area": max(0.0, (x2 - x1) * (y2 - y1)),
                            "vehicle_conf": float(box.conf[0]),
                        })
        idx += 1
    cap.release()

    if not tracks:
        notes.append("No vehicles tracked in sampled frames.")
        out = VisionOutput(
            clip_name=video_file.name, duration_s=duration_s, fps=round(fps, 3),
            frame_size=(width, height), sampled_fps=round(fps / stride, 3), track_id=-1,
            target_selection_reason="no vehicle detected", vehicle_class="car",
            vehicle_crop_path=None, frames=[], notes=notes,
        )
        (run_path / "vision.json").write_text(out.model_dump_json(indent=2), encoding="utf-8")
        return out

    # Target = longest track, tie-break by median box area
    best_track = max(
        tracks.values(),
        key=lambda t: (len(t["observations"]), float(np.median([o["area"] for o in t["observations"]]))),
    )
    target_id = best_track["track_id"]
    target_class = Counter(best_track["classes"]).most_common(1)[0][0]
    observations = best_track["observations"]
    selection_reason = f"longest track ({len(observations)} frames), largest median vehicle box"
    if len(tracks) > 1:
        notes.append(f"Detected {len(tracks)} vehicle tracks; selected track_id={target_id}.")

    best_obs = max(observations, key=lambda o: o["area"] * o["vehicle_conf"])
    obs_by_idx = {o["frame_idx"]: o for o in observations}
    last_needed = max(obs_by_idx)

    # ---------------- Pass 2: re-read ONLY the target's frames ----------------
    records: List[dict] = []
    vehicle_crop_rel: Optional[str] = None

    cap = cv2.VideoCapture(str(video_file))
    idx = 0
    while idx <= last_needed:
        if not cap.grab():
            break
        if idx in obs_by_idx:
            ok, frame = cap.retrieve()
            if ok:
                obs = obs_by_idx[idx]
                vbox = obs["vehicle_box"]

                if idx == best_obs["frame_idx"]:
                    bx1, by1, bx2, by2 = map(int, vbox)
                    vcrop = frame[max(0, by1):min(height, by2), max(0, bx1):min(width, bx2)]
                    if vcrop.size > 0:
                        vehicle_crop_rel = f"frames/vehicle_t{target_id}.jpg"
                        cv2.imwrite(str(run_path / vehicle_crop_rel), vcrop)

                rec = {
                    "frame_idx": idx, "timestamp_s": obs["timestamp_s"], "vehicle_box": vbox,
                    "sharpness": 0.0, "used": False, "skip_reason": None,
                    "plate_box": None, "plate_rel": None, "enh_rel": None,
                    "crop": None, "raw_text": "", "chars": [],
                }
                crop, pbox, reason = _detect_plate(lp_model, frame, vbox)
                if crop is None:
                    rec["skip_reason"] = reason
                else:
                    rec["crop"] = crop
                    rec["plate_box"] = pbox
                    rec["sharpness"] = round(_sharpness(crop), 2)
                    rec["plate_rel"] = f"frames/plate_t{target_id}_f{idx}.jpg"
                    cv2.imwrite(str(run_path / rec["plate_rel"]), crop)
                records.append(rec)
        idx += 1
    cap.release()

    # ---------------- Choose which frames get OCR ----------------
    detected = [r for r in records if r["crop"] is not None]
    if not detected:
        notes.append("Plate was not detected in any frame of the target vehicle.")

    passing = [r for r in detected if r["sharpness"] >= settings.min_frame_sharpness]
    for r in detected:
        if r not in passing:
            r["skip_reason"] = (
                f"blurry plate (sharpness {r['sharpness']} < {settings.min_frame_sharpness})"
            )
    if detected and not passing:
        passing = sorted(detected, key=lambda r: r["sharpness"], reverse=True)[:FALLBACK_FRAMES]
        for r in passing:
            r["skip_reason"] = None
        notes.append(
            f"No frame passed the sharpness gate ({settings.min_frame_sharpness}); "
            f"the {len(passing)} sharpest frames were used instead. Treat the result as low quality."
        )

    passing.sort(key=lambda r: r["sharpness"], reverse=True)
    for r in passing[MAX_OCR_FRAMES:]:
        r["skip_reason"] = f"not among the {MAX_OCR_FRAMES} sharpest plate frames"
    selected = passing[:MAX_OCR_FRAMES]

    for r in selected:
        enhanced = _enhance_plate(r["crop"])
        r["enh_rel"] = f"frames/enh_t{target_id}_f{r['frame_idx']}.jpg"
        cv2.imwrite(str(run_path / r["enh_rel"]), enhanced)
        raw_text, chars = _read_plate_chars(ocr_reader, enhanced)
        if not raw_text:
            r["skip_reason"] = "OCR returned empty text"
            continue
        r["raw_text"], r["chars"], r["used"] = raw_text, chars, True

    # ---------------- Quality weights (sharpness only, sum to 1) ----------------
    total = sum(r["sharpness"] for r in records if r["used"])
    frame_readings: List[FrameReading] = []
    for r in sorted(records, key=lambda r: r["frame_idx"]):
        weight = round(r["sharpness"] / total, 4) if (r["used"] and total > 0) else 0.0
        frame_readings.append(FrameReading(
            frame_idx=r["frame_idx"], timestamp_s=r["timestamp_s"], sharpness=r["sharpness"],
            quality_weight=weight, used=r["used"], skip_reason=r["skip_reason"],
            vehicle_box=r["vehicle_box"], plate_box=r["plate_box"],
            plate_crop_path=r["plate_rel"], enhanced_crop_path=r["enh_rel"],
            raw_text=r["raw_text"], chars=r["chars"],
        ))

    out = VisionOutput(
        clip_name=video_file.name, duration_s=duration_s, fps=round(fps, 3),
        frame_size=(width, height), sampled_fps=round(fps / stride, 3),
        track_id=target_id, target_selection_reason=selection_reason,
        vehicle_class=target_class, vehicle_crop_path=vehicle_crop_rel,
        frames=frame_readings, notes=notes,
    )
    (run_path / "vision.json").write_text(out.model_dump_json(indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the vision pipeline on one clip.")
    ap.add_argument("video", nargs="?", default="data/clips/clip1.mp4")
    ap.add_argument("run_dir", nargs="?", default="runs/test_run_01")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    args = ap.parse_args()

    result = run_vision(args.video, Settings(sample_fps=args.sample_fps), args.run_dir)
    for f in result.frames:
        status = "USED " if f.used else "skip "
        print(f"{status} f{f.frame_idx:<5} sharp={f.sharpness:<8} w={f.quality_weight:<6} "
              f"text={f.raw_text or '-':<12} {f.skip_reason or ''}")
    print("notes:", result.notes)
    print(f"\nWrote {Path(args.run_dir) / 'vision.json'}")
