"""
smoke_test.py - Phase 0 offline check. Run with Wi-Fi OFF to prove nothing downloads.

    python smoke_test.py [path/to/clip.mp4]
"""

import sys
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

from vision.pipeline import _order_results  # same line-aware ordering the pipeline uses

CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/clips/clip1.mp4"
REQUIRED = ["models/yolov8n.pt", "models/yolov8s.pt", "models/lp_detector.pt", "models/easyocr"]
PAD = 0.08


def main() -> None:
    # 0. Fail early and clearly if something is missing
    missing = [p for p in REQUIRED if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Missing: {missing}. Download weights while online first.")
    if not Path(CLIP).exists():
        raise SystemExit(f"Clip not found: {CLIP}")

    # 1. Middle frame of the clip
    cap = cv2.VideoCapture(CLIP)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {CLIP}")
    h, w = frame.shape[:2]

    # 2. Models, strictly offline
    veh_model = YOLO("models/yolov8n.pt")
    lp_model = YOLO("models/lp_detector.pt")
    reader = easyocr.Reader(["en"], gpu=False, model_storage_directory="models/easyocr",
                            download_enabled=False)

    # 3. Vehicles (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
    veh_res = veh_model.predict(frame, classes=[2, 3, 5, 7], conf=0.35, verbose=False)[0]
    print(f"Vehicles detected: {len(veh_res.boxes)}")

    # 4. Plates + OCR
    plates = 0
    for box in veh_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        veh = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if veh.size == 0:
            continue
        for lp in lp_model.predict(veh, conf=0.3, verbose=False)[0].boxes:
            px1, py1, px2, py2 = map(float, lp.xyxy[0])
            dx, dy = (px2 - px1) * PAD, (py2 - py1) * PAD
            crop = veh[max(0, int(py1 - dy)):min(veh.shape[0], int(py2 + dy)),
                       max(0, int(px1 - dx)):min(veh.shape[1], int(px2 + dx))]
            if crop.size == 0:
                continue
            plates += 1
            cv2.imwrite(f"smoke_plate_{plates}.jpg", crop)
            ocr = _order_results(list(reader.readtext(crop, detail=1)))
            print(f"Plate #{plates}:", [(t[1], round(float(t[2]), 2)) for t in ocr])

    if plates == 0:
        print("No plate on this frame, but all 3 models loaded and ran offline OK.")
    else:
        print(f"OK: {plates} plate(s) read offline.")


if __name__ == "__main__":
    main()
