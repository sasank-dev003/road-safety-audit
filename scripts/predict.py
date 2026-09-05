"""
predict.py
----------
YOLOv11 Segmentation — Video Inference
AI Automated Road Safety Audit Project

- Segmentation classes  → colored translucent masks (retina-style overlay)
- Sign classes          → bounding boxes only
- CENTRE_LANE           → per-dash masks merged into one continuous overlay
- Output saved as a new video file alongside the input
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  USER CONFIGURATION
# ─────────────────────────────────────────────

# Path to your trained weights
# Set via environment variable or edit directly:
#   export RSA_WEIGHTS=/path/to/best.pt
WEIGHTS = os.environ.get("RSA_WEIGHTS", r"C:\path\to\best.pt")

# Input video
# Set via environment variable or edit directly:
#   export RSA_INPUT_VIDEO=/path/to/input.mp4
INPUT_VIDEO = os.environ.get("RSA_INPUT_VIDEO", r"C:\path\to\input.mp4")
# Output video (leave empty "" to auto-name alongside input)
OUTPUT_VIDEO = ""

# Confidence threshold (0.0 – 1.0)
CONF_THRESHOLD = 0.35

# IoU threshold for NMS
IOU_THRESHOLD = 0.45

# Show live preview while processing (set False for headless/Kaggle)
SHOW_PREVIEW = False

# ─────────────────────────────────────────────
#  CENTRE LANE FILL CONFIG
#  Morphological close kernel to bridge gaps between dashes.
#  Increase CENTRE_LANE_FILL_KERNEL if dashes are far apart.
# ─────────────────────────────────────────────

CENTRE_LANE_FILL        = True   # set False to revert to per-dash masks
CENTRE_LANE_FILL_KERNEL = 60     # px — controls how far gaps are bridged

# ─────────────────────────────────────────────
#  CLASS DISPLAY CONFIG
#  "type" : "mask" → translucent colored overlay
#           "box"  → bounding box only
#  "color": BGR tuple
#  "alpha": mask transparency (0.0 = invisible, 1.0 = opaque) — only for masks
# ─────────────────────────────────────────────

CLASS_CONFIG = {
    #  class name (must match data.yaml order exactly)     type      color (BGR)          alpha
    "SHOULDER_LANE"        : {"type": "mask", "color": (128,   0, 128), "alpha": 0.45},  # purple
    "CENTRE_LANE"          : {"type": "mask", "color": (  0, 165, 255), "alpha": 0.45},  # orange
    "KERB"                 : {"type": "mask", "color": (144, 238, 144), "alpha": 0.45},  # light green
    "RIGID_CRASH_BARRIER"  : {"type": "mask", "color": (  0,  80,   0), "alpha": 0.45},  # dark green
    "W-BEAM_CRASH_BARRIER" : {"type": "mask", "color": (255, 255,   0), "alpha": 0.45},  # cyan
    "CHEVRON_SIGNS"        : {"type": "box",  "color": (  0, 200, 255), "alpha": 1.0 },  # yellow-ish
    "INFORMATIVE_SIGNS"    : {"type": "box",  "color": (255,   0,   0), "alpha": 1.0 },  # blue
    "MANDATORY_SIGNS"      : {"type": "box",  "color": (  0,   0, 255), "alpha": 1.0 },  # red
    "WARNING_SIGNS"        : {"type": "box",  "color": (  0, 128, 255), "alpha": 1.0 },  # orange-red
}

# Label text settings
SHOW_LABELS     = True
SHOW_CONF       = True
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.55
FONT_THICKNESS  = 1
LABEL_BG_ALPHA  = 0.6   # background pill behind label text

# Bounding box line thickness
BOX_THICKNESS = 2

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

def draw_mask(frame, mask_xy, color, alpha):
    """Draw a translucent filled polygon mask on the frame."""
    overlay = frame.copy()
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # thin border for crispness
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)


def draw_mask_from_bitmap(frame, mask_bitmap, color, alpha):
    """Draw a translucent overlay from a binary bitmap mask (for merged centre lane)."""
    overlay = frame.copy()
    overlay[mask_bitmap > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # draw contour border
    contours, _ = cv2.findContours(mask_bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1)


def merge_centre_lane_masks(mask_xy_list, height, width, kernel_size):
    """
    Takes a list of per-dash polygon arrays, rasterizes them all onto one
    binary canvas, then morphologically closes the gaps between dashes.
    Returns a uint8 binary bitmap (same size as frame).
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    for mask_xy in mask_xy_list:
        pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
    return filled


def draw_box(frame, x1, y1, x2, y2, color, thickness=BOX_THICKNESS):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, x, y, color):
    """Draw label with a semi-transparent background pill."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), color, -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, frame, 1 - LABEL_BG_ALPHA, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def process_video(weights, input_path, output_path, conf, iou):
    model = YOLO(weights)
    class_names = model.names  # {0: 'SHOULDER_LANE', 1: 'CENTRE_LANE', ...}

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Input  : {input_path}")
    print(f"[INFO] Output : {output_path}")
    print(f"[INFO] Frames : {total}  |  FPS: {fps:.1f}  |  Res: {width}x{height}")
    print(f"[INFO] Classes: {list(class_names.values())}")
    print(f"[INFO] Centre lane fill: {'ON' if CENTRE_LANE_FILL else 'OFF'}"
          + (f" (kernel={CENTRE_LANE_FILL_KERNEL}px)" if CENTRE_LANE_FILL else "") + "\n")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            verbose=False,
            retina_masks=True,   # full-res masks instead of downscaled
        )

        for result in results:
            boxes = result.boxes
            masks = result.masks  # None if no mask predictions

            if boxes is None:
                continue

            # ── Collect centre lane dash masks for this frame ──
            centre_lane_masks   = []   # list of mask_xy arrays
            centre_lane_cfg     = CLASS_CONFIG.get("CENTRE_LANE", {"color": (0, 165, 255), "alpha": 0.45})
            centre_lane_label_pos = None   # top-left of first dash bbox for label

            for i, box in enumerate(boxes):
                cls_id   = int(box.cls[0])
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0])

                cfg   = CLASS_CONFIG.get(cls_name, {"type": "mask", "color": (200, 200, 200), "alpha": 0.4})
                color = cfg["color"]
                alpha = cfg["alpha"]
                mode  = cfg["type"]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── Centre lane: collect dashes, draw later ──
                if cls_name == "CENTRE_LANE" and CENTRE_LANE_FILL:
                    if masks is not None and i < len(masks.xy):
                        mask_pts = masks.xy[i]
                        if len(mask_pts) >= 3:
                            centre_lane_masks.append(mask_pts)
                    if centre_lane_label_pos is None:
                        centre_lane_label_pos = (x1, y1)
                    continue   # skip normal drawing for this detection

                # ── All other classes draw normally ──
                if mode == "mask" and masks is not None and i < len(masks.xy):
                    mask_pts = masks.xy[i]
                    if len(mask_pts) >= 3:
                        draw_mask(frame, mask_pts, color, alpha)

                elif mode == "box":
                    draw_box(frame, x1, y1, x2, y2, color)

                if SHOW_LABELS:
                    label = cls_name.replace("_", " ")
                    if SHOW_CONF:
                        label += f" {conf_val:.2f}"
                    draw_label(frame, label, x1, y1 - 2, color)

            # ── Draw merged centre lane ──
            if centre_lane_masks:
                merged = merge_centre_lane_masks(
                    centre_lane_masks, height, width, CENTRE_LANE_FILL_KERNEL
                )
                draw_mask_from_bitmap(
                    frame, merged,
                    centre_lane_cfg["color"],
                    centre_lane_cfg["alpha"]
                )
                if SHOW_LABELS and centre_lane_label_pos is not None:
                    draw_label(frame, "CENTRE LANE", *centre_lane_label_pos,
                               centre_lane_cfg["color"])

        writer.write(frame)

        if SHOW_PREVIEW:
            cv2.imshow("Road Safety — Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit by user.")
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total} frames...", end="\r")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Saved to: {output_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    input_path = Path(INPUT_VIDEO)
    if not input_path.exists():
        print(f"[ERROR] Video not found: {INPUT_VIDEO}")
        raise SystemExit

    if OUTPUT_VIDEO:
        output_path = str(OUTPUT_VIDEO)
    else:
        output_path = str(input_path.parent / (input_path.stem + "_predicted" + input_path.suffix))

    process_video(
        weights     = WEIGHTS,
        input_path  = str(input_path),
        output_path = output_path,
        conf        = CONF_THRESHOLD,
        iou         = IOU_THRESHOLD,
    )
