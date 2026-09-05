"""
verify_labels.py
-----------------
Verifies YOLO segmentation label .txt files (produced by mask_to_yolo.py)
by rendering them back onto the original frames, and computes quality
statistics + per-frame anomaly flags — so annotation problems get caught
BEFORE a training run, not after a bad mAP.

Part of the AI Automated Road Safety Audit project.

Output:
    verify_output/
        annotated/               <-- rendered sample of frames with labels drawn
        verification_log.txt     <-- full run log + stats + flagged frame list
        flagged_frames.txt       <-- plain list of flagged frame stems (copy-paste into AE)
"""

import os
import cv2
import random
import logging
import numpy as np
import statistics
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURATION — edit before running
# ─────────────────────────────────────────

# Set via environment variables or edit directly:
#   export RSA_IMAGES_DIR=/path/to/original_frames
#   export RSA_LABELS_DIR=/path/to/labels
#   export RSA_VERIFY_OUTPUT=/path/to/verify_output
IMAGES_DIR = os.environ.get("RSA_IMAGES_DIR", r"C:\path\to\original_frames")
LABELS_DIR = os.environ.get("RSA_LABELS_DIR", r"C:\path\to\labels")
OUTPUT_DIR = os.environ.get("RSA_VERIFY_OUTPUT", r"C:\path\to\verify_output")

IMAGE_EXT = ".png"   # original_frames are lossless PNG — verify pre-merge, pre-JPEG-conversion

# class_id -> class name.  MUST match the id scheme mask_to_yolo.py used
# when it wrote these particular label files (check CLASS_COLORS keys for that run).
CLASS_ID_TO_NAME = {
    0: "SHOULDER_LANE",
    1: "CENTRE_LANE",
    2: "KERB",
    3: "RIGID_CRASH_BARRIER",
    4: "W-BEAM_CRASH_BARRIER",
    5: "CHEVRON_SIGNS",
    6: "MANDATORY_SIGNS",
    7: "WARNING_SIGNS",
    8: "INFORMATIVE_SIGNS",
}

# Display style per class — mirrors predict.py's CLASS_CONFIG so verification
# renders look like inference renders.
CLASS_DISPLAY = {
    "SHOULDER_LANE"        : {"type": "mask", "color": (128,   0, 128), "alpha": 0.45},  # purple
    "CENTRE_LANE"           : {"type": "mask", "color": (  0, 165, 255), "alpha": 0.45},  # orange
    "KERB"                  : {"type": "mask", "color": (144, 238, 144), "alpha": 0.45},  # light green
    "RIGID_CRASH_BARRIER"   : {"type": "mask", "color": (  0,  80,   0), "alpha": 0.45},  # dark green
    "W-BEAM_CRASH_BARRIER"  : {"type": "mask", "color": (255, 255,   0), "alpha": 0.45},  # cyan
    "CHEVRON_SIGNS"         : {"type": "box",  "color": (  0, 200, 100), "alpha": 1.0 },  # distinct from WARNING
    "MANDATORY_SIGNS"       : {"type": "box",  "color": (  0,   0, 255), "alpha": 1.0 },  # red
    "WARNING_SIGNS"         : {"type": "box",  "color": (  0, 128, 255), "alpha": 1.0 },  # orange-red
    "INFORMATIVE_SIGNS"     : {"type": "box",  "color": (255,   0,   0), "alpha": 1.0 },  # blue
}

# Per-class minimum contour area used when THESE labels were generated.
# Keep in sync with mask_to_yolo.py's MIN_CONTOUR_AREA / POLYGON_CONFIG for this run —
# used only for the "near noise-cutoff" flag below, not for filtering anything here.
MIN_CONTOUR_AREA = {
    "SHOULDER_LANE"        : 100,
    "CENTRE_LANE"           : 50,
    "KERB"                  : 50,
    "RIGID_CRASH_BARRIER"   : 300,
    "W-BEAM_CRASH_BARRIER"  : 300,
    "CHEVRON_SIGNS"         : 30,
    "MANDATORY_SIGNS"       : 30,
    "WARNING_SIGNS"         : 30,
    "INFORMATIVE_SIGNS"     : 30,
}
DEFAULT_MIN_AREA = 100

# Sampling — rendering every frame is slow; stats are ALWAYS computed on every frame.
SAMPLE_MODE = "every_nth"      # "all" | "every_nth" | "random_n"
SAMPLE_N    = 20               # every_nth -> render 1 in N frames | random_n -> render N frames total
SAMPLE_SEED = 42

# Anomaly thresholds
OUTLIER_STDDEV_THRESHOLD    = 2.5   # per-class instance-count-per-frame outlier (fragment spikes)
OVERLAP_IOU_THRESHOLD       = 0.15  # SHOULDER_LANE / CENTRE_LANE overlap flag
NEAR_THRESHOLD_MARGIN       = 1.5   # "near min-area" = within 1.5x of MIN_CONTOUR_AREA

# Rendering
FONT              = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE        = 0.5
FONT_THICKNESS    = 1
SUMMARY_BG_ALPHA  = 0.6
SUMMARY_BG_COLOR  = (0, 0, 0)

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "verification_log.txt"))
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# PARSING & VALIDATION HELPERS
# ─────────────────────────────────────────

def validate_sync(images_dir, labels_dir):
    """Same convention as mask_to_yolo.py — match by stem, log mismatches, never crash."""
    image_stems = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(IMAGE_EXT)}
    label_stems = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}

    matched      = sorted(image_stems & label_stems)
    missing_lbl  = sorted(image_stems - label_stems)
    missing_img  = sorted(label_stems - image_stems)
    return matched, missing_lbl, missing_img


def parse_label_file(label_path, img_w, img_h):
    """
    Parses one YOLO label .txt file.
    Returns:
        instances: list of dicts:
            {class_id, class_name, points_px (Nx2 int array), area_px, num_points}
        line_flags: list of strings describing malformed lines found
    """
    instances  = []
    line_flags = []

    with open(label_path, "r") as f:
        raw_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for li, line in enumerate(raw_lines):
        parts = line.split()

        try:
            class_id = int(parts[0])
        except (ValueError, IndexError):
            line_flags.append(f"line {li}: unparsable class_id")
            continue

        coords = parts[1:]
        if len(coords) < 6 or len(coords) % 2 != 0:
            line_flags.append(f"line {li}: invalid coordinate count ({len(coords)}) — need even, >=6")
            continue

        try:
            vals = [float(c) for c in coords]
        except ValueError:
            line_flags.append(f"line {li}: non-numeric coordinate")
            continue

        if any(v < 0.0 or v > 1.0 for v in vals):
            line_flags.append(f"line {li}: coordinate outside [0.0, 1.0] range")
            # still keep going — clamp isn't our job, just flag and use as-is

        if class_id not in CLASS_ID_TO_NAME:
            line_flags.append(f"line {li}: class_id {class_id} not in CLASS_ID_TO_NAME")
            continue

        class_name = CLASS_ID_TO_NAME[class_id]

        pts_norm = np.array(vals, dtype=np.float64).reshape(-1, 2)
        pts_px   = pts_norm.copy()
        pts_px[:, 0] *= img_w
        pts_px[:, 1] *= img_h
        pts_px = pts_px.astype(np.int32)

        area_px = float(cv2.contourArea(pts_px))

        instances.append({
            "class_id"  : class_id,
            "class_name": class_name,
            "points_px" : pts_px,
            "area_px"   : area_px,
            "num_points": len(pts_px),
        })

    return instances, line_flags


def rasterize(points_px_list, height, width):
    """Rasterizes a list of polygons (each Nx2 int array) onto one binary canvas."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    for pts in points_px_list:
        cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 255)
    return canvas


def compute_iou(mask_a, mask_b):
    """IoU between two binary uint8 masks (0/255)."""
    inter = np.count_nonzero(cv2.bitwise_and(mask_a, mask_b))
    union = np.count_nonzero(cv2.bitwise_or(mask_a, mask_b))
    if union == 0:
        return 0.0
    return inter / union


# ─────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────

def draw_mask(frame, pts_px, color, alpha):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts_px.reshape(-1, 1, 2)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts_px.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=1)


def draw_box_from_polygon(frame, pts_px, color):
    x, y, w, h = cv2.boundingRect(pts_px.reshape(-1, 1, 2))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return x, y


def draw_label(frame, text, x, y, color):
    (tw, th), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (x, y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)


def draw_frame_summary(frame, summary_text):
    (tw, th), baseline = cv2.getTextSize(summary_text, FONT, FONT_SCALE, FONT_THICKNESS)
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + baseline + pad * 2), SUMMARY_BG_COLOR, -1)
    cv2.addWeighted(overlay, SUMMARY_BG_ALPHA, frame, 1 - SUMMARY_BG_ALPHA, 0, frame)
    cv2.putText(frame, summary_text, (pad, th + pad), FONT, FONT_SCALE, (255, 255, 255),
                FONT_THICKNESS, cv2.LINE_AA)


def render_frame(image_path, instances, out_path):
    frame = cv2.imread(image_path)
    if frame is None:
        log.error(f"Could not read image for rendering: {image_path}")
        return

    per_class_count = defaultdict(int)
    for inst in instances:
        cfg = CLASS_DISPLAY.get(inst["class_name"], {"type": "mask", "color": (200, 200, 200), "alpha": 0.4})
        color, mode, alpha = cfg["color"], cfg["type"], cfg["alpha"]
        per_class_count[inst["class_name"]] += 1

        if mode == "mask":
            draw_mask(frame, inst["points_px"], color, alpha)
            x, y = inst["points_px"][0]
        else:
            x, y = draw_box_from_polygon(frame, inst["points_px"], color)

        draw_label(frame, inst["class_name"].replace("_", " "), int(x), max(15, int(y) - 4), color)

    total = len(instances)
    breakdown = " ".join(f"{name}:{cnt}" for name, cnt in sorted(per_class_count.items()))
    summary = f"{total} objects" + (f" — {breakdown}" if breakdown else " — EMPTY FRAME")
    draw_frame_summary(frame, summary)

    cv2.imwrite(out_path, frame)


# ─────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────

def choose_render_stems(matched_stems, mode, n, seed):
    if mode == "all":
        return set(matched_stems)
    if mode == "every_nth":
        return set(matched_stems[::max(1, n)])
    if mode == "random_n":
        rng = random.Random(seed)
        return set(rng.sample(matched_stems, min(n, len(matched_stems))))
    log.warning(f"Unknown SAMPLE_MODE '{mode}', defaulting to 'every_nth' with N=20")
    return set(matched_stems[::20])


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("STEP 1: Validating sync between images and labels")
    log.info("=" * 60)

    matched, missing_lbl, missing_img = validate_sync(IMAGES_DIR, LABELS_DIR)
    log.info(f"  Matched pairs       : {len(matched)}")
    log.info(f"  Images without label: {len(missing_lbl)}")
    log.info(f"  Labels without image: {len(missing_img)}")

    if missing_lbl:
        log.warning("Images with NO matching label file (skipped):")
        for s in missing_lbl:
            log.warning(f"  {s}{IMAGE_EXT}")
    if missing_img:
        log.warning("Label files with NO matching image (skipped):")
        for s in missing_img:
            log.warning(f"  {s}.txt")

    if not matched:
        log.error("No matched pairs found. Check IMAGES_DIR / LABELS_DIR.")
        return

    render_stems = choose_render_stems(matched, SAMPLE_MODE, SAMPLE_N, SAMPLE_SEED)
    log.info(f"  Sample mode: {SAMPLE_MODE}  ->  rendering {len(render_stems)} / {len(matched)} frames")

    log.info("")
    log.info("=" * 60)
    log.info("STEP 2: Parsing all label files")
    log.info("=" * 60)

    frame_data      = {}   # stem -> list of instance dicts
    frame_flags     = defaultdict(list)   # stem -> list of flag strings
    frame_dims      = {}   # stem -> (w, h)
    per_class_counts     = defaultdict(list)  # class_name -> [count_per_frame_where_present]
    per_class_points     = defaultdict(list)  # class_name -> [num_points, ...]
    per_class_areas      = defaultdict(list)  # class_name -> [area_px, ...]
    per_class_total      = defaultdict(int)
    per_class_frame_hits = defaultdict(int)   # frames containing >=1 instance of this class

    for idx, stem in enumerate(matched, 1):
        img_path = os.path.join(IMAGES_DIR, stem + IMAGE_EXT)
        lbl_path = os.path.join(LABELS_DIR, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            log.error(f"Could not read image: {img_path} — skipping frame")
            continue
        h, w = img.shape[:2]
        frame_dims[stem] = (w, h)

        instances, line_flags = parse_label_file(lbl_path, w, h)
        frame_data[stem] = instances
        for lf in line_flags:
            frame_flags[stem].append(f"malformed label: {lf}")

        if not instances:
            frame_flags[stem].append("empty label file (0 valid instances)")

        per_frame_class_count = defaultdict(int)
        for inst in instances:
            cname = inst["class_name"]
            per_class_total[cname] += 1
            per_class_points[cname].append(inst["num_points"])
            per_class_areas[cname].append(inst["area_px"])
            per_frame_class_count[cname] += 1

            min_area = MIN_CONTOUR_AREA.get(cname, DEFAULT_MIN_AREA)
            if min_area <= inst["area_px"] <= min_area * NEAR_THRESHOLD_MARGIN:
                frame_flags[stem].append(
                    f"{cname}: instance area {inst['area_px']:.0f}px near MIN_CONTOUR_AREA "
                    f"({min_area}px) — verify it's not noise"
                )

        for cname, cnt in per_frame_class_count.items():
            per_class_counts[cname].append(cnt)
            per_class_frame_hits[cname] += 1

        if idx % 200 == 0:
            log.info(f"  Parsed {idx}/{len(matched)} label files...")

    log.info(f"  Parsed {len(matched)}/{len(matched)} label files.")

    # ── Per-class outlier check (fragment spikes) — needs full stats first ──
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3: Flagging per-frame anomalies")
    log.info("=" * 60)

    class_mean_std = {}
    for cname, counts in per_class_counts.items():
        if len(counts) >= 2:
            mean = statistics.mean(counts)
            std  = statistics.pstdev(counts)
        else:
            mean, std = (counts[0], 0.0) if counts else (0.0, 0.0)
        class_mean_std[cname] = (mean, std)

    for stem, instances in frame_data.items():
        per_frame_class_count = defaultdict(int)
        for inst in instances:
            per_frame_class_count[inst["class_name"]] += 1

        for cname, cnt in per_frame_class_count.items():
            mean, std = class_mean_std.get(cname, (0.0, 0.0))
            if std > 0 and cnt > mean + OUTLIER_STDDEV_THRESHOLD * std:
                frame_flags[stem].append(
                    f"{cname}: instance count {cnt} is an outlier "
                    f"(mean={mean:.1f}, std={std:.1f}) — possible propagation artifact"
                )

        # SHOULDER_LANE / CENTRE_LANE overlap check
        w, h = frame_dims[stem]
        shoulder_pts = [i["points_px"] for i in instances if i["class_name"] == "SHOULDER_LANE"]
        centre_pts   = [i["points_px"] for i in instances if i["class_name"] == "CENTRE_LANE"]
        if shoulder_pts and centre_pts:
            mask_a = rasterize(shoulder_pts, h, w)
            mask_b = rasterize(centre_pts, h, w)
            iou = compute_iou(mask_a, mask_b)
            if iou > OVERLAP_IOU_THRESHOLD:
                frame_flags[stem].append(
                    f"SHOULDER_LANE/CENTRE_LANE overlap IoU={iou:.2f} "
                    f"(> {OVERLAP_IOU_THRESHOLD}) — check for over-segmentation"
                )

    flagged_stems = {s: fl for s, fl in frame_flags.items() if fl}
    log.info(f"  Frames flagged: {len(flagged_stems)} / {len(matched)}")

    # ── Render sampled frames ──
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4: Rendering annotated sample")
    log.info("=" * 60)

    rendered = 0
    for stem in matched:
        if stem not in render_stems:
            continue
        img_path = os.path.join(IMAGES_DIR, stem + IMAGE_EXT)
        out_path = os.path.join(ANNOTATED_DIR, stem + "_verify.jpg")
        render_frame(img_path, frame_data.get(stem, []), out_path)
        rendered += 1

    log.info(f"  Rendered {rendered} annotated images -> {ANNOTATED_DIR}")

    # ── Per-class summary table ──
    log.info("")
    log.info("=" * 60)
    log.info("CLASS SUMMARY")
    log.info("=" * 60)

    for cid, cname in sorted(CLASS_ID_TO_NAME.items()):
        total = per_class_total.get(cname, 0)
        if total == 0:
            log.warning(f"  {cname:<22s}: 0 instances — check color mapping / AE export for this class")
            continue

        pts   = per_class_points[cname]
        areas = per_class_areas[cname]
        frame_pct = 100.0 * per_class_frame_hits[cname] / len(matched)

        log.info(
            f"  {cname:<22s}: {total:5d} instances | "
            f"present in {per_class_frame_hits[cname]:5d} frames ({frame_pct:5.1f}%) | "
            f"points avg/min/max = {statistics.mean(pts):.1f}/{min(pts)}/{max(pts)} | "
            f"area avg/min/max px = {statistics.mean(areas):.0f}/{min(areas):.0f}/{max(areas):.0f}"
        )

    # ── Flagged frames, worst first ──
    log.info("")
    log.info("=" * 60)
    log.info(f"FLAGGED FRAMES ({len(flagged_stems)} total, sorted by severity)")
    log.info("=" * 60)

    ranked = sorted(flagged_stems.items(), key=lambda kv: len(kv[1]), reverse=True)
    for stem, flags in ranked:
        log.info(f"  {stem}  ({len(flags)} flag(s))")
        for f in flags:
            log.info(f"      - {f}")

    flagged_path = os.path.join(OUTPUT_DIR, "flagged_frames.txt")
    with open(flagged_path, "w") as f:
        f.write("\n".join(stem for stem, _ in ranked))
    log.info("")
    log.info(f"[SAVED] Flagged frame list -> {flagged_path}")
    log.info(f"[SAVED] Full log           -> {os.path.join(OUTPUT_DIR, 'verification_log.txt')}")
    log.info(f"[SAVED] Annotated sample   -> {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()
