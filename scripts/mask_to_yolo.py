"""
mask_to_yolo.py — Convert color-coded segmentation masks to YOLO segmentation labels.

This script processes mask images, extracts polygons per class color, and generates
YOLO-format label files with train/val/test splitting.

Part of the AI Automated Road Safety Audit project.
"""

import os
import sys
import logging
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np

# =============================================================================
# CONFIGURATION — Edit these values before running
# =============================================================================

# Input/Output paths
# INPUT_ROOT must contain original_frames/ and mask_frames/ for ONE video.
# Set via environment variable or edit directly:
#   export RSA_INPUT_ROOT=/path/to/video_dataset
#   export RSA_OUTPUT_ROOT=/path/to/output
INPUT_ROOT = Path(os.environ.get("RSA_INPUT_ROOT", r"C:\path\to\your\video_dataset"))
OUTPUT_ROOT = Path(os.environ.get("RSA_OUTPUT_ROOT", INPUT_ROOT / "split"))  # creates split/train, split/valid, split/test + data.yaml here
LOG_FILE = OUTPUT_ROOT / "conversion_log.txt"

# Original frames and mask frames are exported in DIFFERENT formats — set both.
ORIGINAL_IMAGE_EXT = ".jpg"   # extension used in original_frames/  (change to ".jpeg" if that's what AE exports)
MASK_IMAGE_EXT      = ".png"   # extension used in mask_frames/ (must stay lossless PNG for exact color match)

# Color tolerance for mask matching (0 = exact match, higher = more forgiving)
COLOR_TOLERANCE = 0

# Train/Valid/Test split ratios (must sum to 1.0)
# NOTE: this is a per-video run, so this is a plain frame-level random split —
# NOT video-grouped (video-aware grouping belongs in merge_datasets.py once
# multiple videos are combined, not here where there's only one video anyway).
SPLIT_RATIO = {"train": 0.80, "valid": 0.10, "test": 0.10}

# Random seed for reproducible splits
RANDOM_SEED = 42

# Class → Color mapping (BGR format, OpenCV loads as BGR)
CLASS_COLORS = {
    0: {"name": "SHOULDER_LANE",         "color": (128, 0, 128),   "enabled": True},   # purple
    1: {"name": "CENTRE_LANE",           "color": (0, 165, 255),   "enabled": True},   # orange
    2: {"name": "KERB",                  "color": (144, 238, 144), "enabled": True},   # light green
    3: {"name": "RIGID_CRASH_BARRIER",   "color": (0, 80, 0),      "enabled": True},   # dark green
    4: {"name": "W-BEAM_CRASH_BARRIER",  "color": (255, 255, 0),   "enabled": True},   # cyan (BGR: 0,255,255)
    5: {"name": "CHEVRON_SIGNS",         "color": (0, 200, 255),   "enabled": True},   # yellow-ish
    6: {"name": "MANDATORY_SIGNS",       "color": (0, 0, 255),     "enabled": True},   # red
    7: {"name": "WARNING_SIGNS",         "color": (0,140,255),            "enabled": True},   # TBD
    8: {"name": "INFORMATIVE_SIGNS",     "color": (30,144,255),            "enabled": True},   # TBD
}

# Per-class polygon tuning settings
POLYGON_CONFIG = {
    "SHOULDER_LANE":        {"epsilon": 2.0, "min_area": 100, "max_points": None},
    "CENTRE_LANE":          {"epsilon": 2.0, "min_area": 50,  "max_points": None},
    "KERB":                 {"epsilon": 1.0, "min_area": 50,  "max_points": None},
    "RIGID_CRASH_BARRIER":  {"epsilon": 3.0, "min_area": 300, "max_points": None},
    "W-BEAM_CRASH_BARRIER": {"epsilon": 3.0, "min_area": 300, "max_points": None},
    "CHEVRON_SIGNS":        {"epsilon": 1.0, "min_area": 30,  "max_points": None},
    "MANDATORY_SIGNS":      {"epsilon": 1.0, "min_area": 30,  "max_points": None},
    "WARNING_SIGNS":        {"epsilon": 1.0, "min_area": 30,  "max_points": None},
    "INFORMATIVE_SIGNS":    {"epsilon": 1.0, "min_area": 30,  "max_points": None},
}

# Fallback defaults if a class is missing from POLYGON_CONFIG
DEFAULT_EPSILON = 2.0
DEFAULT_MIN_AREA = 100
DEFAULT_MAX_POINTS = None

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: Path) -> logging.Logger:
    """Configure logging to output to both console and file."""
    logger = logging.getLogger("mask_to_yolo")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    # File handler (DEBUG and above)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_polygon_config(class_name: str) -> Tuple[float, int, Optional[int]]:
    """Get polygon settings for a class, with fallback to defaults."""
    config = POLYGON_CONFIG.get(class_name, {})
    epsilon = config.get("epsilon", DEFAULT_EPSILON)
    min_area = config.get("min_area", DEFAULT_MIN_AREA)
    max_points = config.get("max_points", DEFAULT_MAX_POINTS)
    return epsilon, min_area, max_points

def color_matches(pixel: np.ndarray, target_color: Tuple[int, int, int], tolerance: int) -> bool:
    """Check if a pixel matches the target color within tolerance."""
    if target_color is None:
        return False
    return all(abs(int(pixel[c]) - target_color[c]) <= tolerance for c in range(3))

def create_color_mask(image: np.ndarray, target_color: Tuple[int, int, int], tolerance: int) -> np.ndarray:
    """Create a binary mask for pixels matching the target color."""
    if target_color is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    if tolerance == 0:
        # Exact match - faster
        lower = np.array(target_color, dtype=np.uint8)
        upper = np.array(target_color, dtype=np.uint8)
        return cv2.inRange(image, lower, upper)
    else:
        # Tolerance matching
        lower = np.array([max(0, c - tolerance) for c in target_color], dtype=np.uint8)
        upper = np.array([min(255, c + tolerance) for c in target_color], dtype=np.uint8)
        return cv2.inRange(image, lower, upper)

def simplify_contour(contour: np.ndarray, epsilon: float, max_points: Optional[int]) -> Optional[np.ndarray]:
    """Simplify a contour to a polygon with optional point cap."""
    if len(contour) < 3:
        return None
    
    # Initial simplification
    simplified = cv2.approxPolyDP(contour, epsilon, True)
    
    # If max_points is set and we exceed it, increase epsilon iteratively
    if max_points is not None and len(simplified) > max_points:
        # Binary search for appropriate epsilon
        low_eps = epsilon
        high_eps = epsilon * 10
        best_simplified = simplified
        
        for _ in range(10):  # Max iterations
            mid_eps = (low_eps + high_eps) / 2
            test_simplified = cv2.approxPolyDP(contour, mid_eps, True)
            
            if len(test_simplified) <= max_points:
                best_simplified = test_simplified
                high_eps = mid_eps
            else:
                low_eps = mid_eps
            
            if len(best_simplified) <= max_points:
                break
        
        simplified = best_simplified
    
    # Ensure minimum 3 points for valid polygon
    if len(simplified) < 3:
        return None
    
    return simplified

def contour_to_normalized_polygon(contour: np.ndarray, img_width: int, img_height: int) -> List[float]:
    """Convert contour to normalized YOLO polygon format [x1, y1, x2, y2, ...]."""
    points = []
    for point in contour.reshape(-1, 2):
        x, y = point
        # Normalize to 0.0-1.0
        nx = x / img_width
        ny = y / img_height
        # Clamp to valid range
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        points.extend([nx, ny])
    return points

def validate_frame_pairs(original_dir: Path, mask_dir: Path, logger: logging.Logger) -> Tuple[Set[str], Set[str]]:
    """Validate that original_frames and mask_frames have matching files (by stem, across their own extensions)."""
    original_files = {f.stem for f in original_dir.glob(f"*{ORIGINAL_IMAGE_EXT}") if f.is_file()}
    mask_files = {f.stem for f in mask_dir.glob(f"*{MASK_IMAGE_EXT}") if f.is_file()}
    
    missing_in_masks = original_files - mask_files
    missing_in_originals = mask_files - original_files
    
    if missing_in_masks:
        logger.warning(f"{len(missing_in_masks)} files in original_frames missing from mask_frames")
        for f in sorted(missing_in_masks)[:10]:
            logger.debug(f"  Missing mask: {f}{MASK_IMAGE_EXT}")
        if len(missing_in_masks) > 10:
            logger.debug(f"  ... and {len(missing_in_masks) - 10} more")
    
    if missing_in_originals:
        logger.warning(f"{len(missing_in_originals)} files in mask_frames missing from original_frames")
        for f in sorted(missing_in_originals)[:10]:
            logger.debug(f"  Missing original: {f}{ORIGINAL_IMAGE_EXT}")
        if len(missing_in_originals) > 10:
            logger.debug(f"  ... and {len(missing_in_originals) - 10} more")
    
    # Return only matched pairs
    matched = original_files & mask_files
    return matched, missing_in_masks | missing_in_originals

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_mask_frame(
    mask_path: Path,
    class_colors: Dict,
    color_tolerance: int,
    logger: logging.Logger
) -> Tuple[List[str], Dict[str, int]]:
    """
    Process a single mask frame and extract YOLO annotations.
    
    Returns:
        - List of annotation lines (one per object)
        - Dict of class_name → instance count for this frame
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        logger.error(f"Failed to load mask: {mask_path}")
        return [], {}
    
    img_height, img_width = mask.shape[:2]
    annotations = []
    class_counts = defaultdict(int)
    
    # Process each enabled class
    for class_id, class_info in class_colors.items():
        if not class_info["enabled"]:
            continue
        
        class_name = class_info["name"]
        target_color = class_info["color"]
        
        if target_color is None:
            logger.debug(f"Skipping class {class_name} (color not set)")
            continue
        
        # Create binary mask for this color
        color_mask = create_color_mask(mask, target_color, color_tolerance)
        
        # Skip if no pixels match
        if cv2.countNonZero(color_mask) == 0:
            continue
        
        # Find contours (RETR_EXTERNAL to avoid nested contours)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get polygon settings for this class
        epsilon, min_area, max_points = get_polygon_config(class_name)
        
        for contour in contours:
            # Check area threshold
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Simplify contour to polygon
            polygon = simplify_contour(contour, epsilon, max_points)
            if polygon is None:
                continue
            
            # Convert to normalized YOLO format
            normalized_points = contour_to_normalized_polygon(polygon, img_width, img_height)
            
            # Build annotation line: class_id x1 y1 x2 y2 ...
            annotation = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
            annotations.append(annotation)
            class_counts[class_name] += 1
    
    return annotations, dict(class_counts)

def process_video_dataset(
    input_root: Path,
    output_root: Path,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Process all frames in a video dataset and generate YOLO labels.
    
    Returns:
        Dict of class_name → total instance count across all frames
    """
    original_dir = input_root / "original_frames"
    mask_dir = input_root / "mask_frames"
    labels_dir = output_root / "labels"
    
    # Validate directories exist
    if not original_dir.exists():
        logger.error(f"Original frames directory not found: {original_dir}")
        return {}
    if not mask_dir.exists():
        logger.error(f"Mask frames directory not found: {mask_dir}")
        return {}
    
    # Validate frame pairs
    matched_files, unmatched_files = validate_frame_pairs(original_dir, mask_dir, logger)
    
    if not matched_files:
        logger.error("No matching frame pairs found between original_frames and mask_frames")
        return {}
    
    # Create output directories
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_class_counts = defaultdict(int)
    total_frames = 0
    empty_frames = 0
    error_frames = 0
    
    # Process each matched frame
    for stem in sorted(matched_files):
        mask_path = mask_dir / f"{stem}{MASK_IMAGE_EXT}"
        label_path = labels_dir / f"{stem}.txt"
        
        try:
            annotations, class_counts = process_mask_frame(
                mask_path, CLASS_COLORS, COLOR_TOLERANCE, logger
            )
            
            # Write label file (even if empty)
            with open(label_path, 'w') as f:
                if annotations:
                    f.write("\n".join(annotations) + "\n")
            
            # Update statistics
            total_frames += 1
            if not annotations:
                empty_frames += 1
            
            for class_name, count in class_counts.items():
                total_class_counts[class_name] += count
            
            logger.debug(f"Processed {stem}.png: {len(annotations)} annotations")
            
        except Exception as e:
            logger.error(f"Error processing {stem}.png: {e}")
            error_frames += 1
    
    logger.info(f"Processed {total_frames} frames ({empty_frames} empty, {error_frames} errors)")
    
    return dict(total_class_counts)

def split_dataset(
    input_root: Path,
    output_root: Path,
    split_ratio: Dict[str, float],
    seed: int,
    logger: logging.Logger
) -> Dict[str, int]:
    """
    Split this video's converted frames into train/valid/test.

    Plain frame-level random split — NOT video-grouped, because this script
    processes ONE video per run (input_root/original_frames + mask_frames).
    Video-aware grouping only matters once multiple videos are combined,
    which happens later in merge_datasets.py, not here.

    Returns:
        Dict of split_name → frame count
    """
    random.seed(seed)

    labels_dir = output_root / "labels"
    original_dir = input_root / "original_frames"

    if not labels_dir.exists():
        logger.error("Labels directory not found. Run process_video_dataset first.")
        return {}

    # Only split frames that have a label file (every processed frame gets one,
    # even if empty, so this is effectively "all processed frames")
    stems = sorted(p.stem for p in labels_dir.glob("*.txt"))
    if not stems:
        logger.error("No label files found for splitting")
        return {}

    random.shuffle(stems)

    total = len(stems)
    train_end = int(total * split_ratio["train"])
    valid_end = train_end + int(total * split_ratio["valid"])

    splits = {
        "train": stems[:train_end],
        "valid": stems[train_end:valid_end],
        "test":  stems[valid_end:],
    }

    split_counts = {"train": 0, "valid": 0, "test": 0}

    for split_name, split_stems in splits.items():
        images_dir = output_root / split_name / "images"
        labels_dir_out = output_root / split_name / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir_out.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # Copy label file
            shutil.copy2(labels_dir / f"{stem}.txt", labels_dir_out / f"{stem}.txt")

            # Copy corresponding original image (already .jpg — no conversion needed)
            image_path = original_dir / f"{stem}{ORIGINAL_IMAGE_EXT}"
            if image_path.exists():
                shutil.copy2(image_path, images_dir / f"{stem}{ORIGINAL_IMAGE_EXT}")
            else:
                logger.warning(f"Image not found for {stem}{ORIGINAL_IMAGE_EXT}")

            split_counts[split_name] += 1

    logger.info(
        f"Split complete: train={split_counts['train']}, "
        f"valid={split_counts['valid']}, test={split_counts['test']}"
    )

    return split_counts


def generate_data_yaml(
    output_root: Path,
    class_colors: Dict,
    logger: logging.Logger
) -> Path:
    """
    Write data.yaml for this split, listing class names in class_id order.

    NOTE: this assumes enabled class_ids are contiguous from 0 (i.e. no
    disabled classes sitting in the middle of the id range). If you disable
    a class in the middle of CLASS_COLORS, the ids in your label files will
    have a gap that this simple name-list won't line up with — re-check
    before training if you ever disable a class again.
    """
    enabled_ids = sorted(cid for cid, info in class_colors.items() if info["enabled"])
    names = [class_colors[cid]["name"] for cid in enabled_ids]

    if enabled_ids != list(range(len(enabled_ids))):
        logger.warning(
            "Enabled class_ids are not contiguous from 0 "
            f"({enabled_ids}) — data.yaml class order will NOT match your "
            "label file class_ids. Fix CLASS_COLORS before training."
        )

    yaml_path = output_root / "data.yaml"
    lines = [
        f"path: {output_root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(names)}",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(names)]

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"data.yaml written to {yaml_path}")
    return yaml_path

def print_class_summary(
    class_counts: Dict[str, int],
    class_colors: Dict,
    logger: logging.Logger
):
    """Print summary table of class instances found."""
    logger.info("=" * 60)
    logger.info("CLASS SUMMARY")
    logger.info("=" * 60)
    
    # Track enabled classes with zero instances for warning
    zero_instance_enabled = []
    
    for class_id in sorted(class_colors.keys()):
        class_info = class_colors[class_id]
        class_name = class_info["name"]
        enabled = class_info["enabled"]
        
        count = class_counts.get(class_name, 0)
        
        if enabled and count == 0:
            zero_instance_enabled.append(class_name)
        
        # Format status
        if not enabled:
            status = "(disabled)"
        elif count == 0:
            status = "(not present — skipped)"
        else:
            status = ""
        
        # Get polygon config used
        epsilon, min_area, max_points = get_polygon_config(class_name)
        config_str = f"ε={epsilon}, min_area={min_area}"
        if max_points:
            config_str += f", max_pts={max_points}"
        
        logger.info(f"  {class_name:<25}: {count:>6} instances {status:<30} [{config_str}]")
    
    logger.info("=" * 60)
    
    # Loud warning for zero-instance enabled classes
    if zero_instance_enabled:
        logger.warning("!" * 60)
        logger.warning("WARNING: The following enabled classes had ZERO instances found!")
        logger.warning("This may indicate a color mismatch or annotation issue:")
        for class_name in zero_instance_enabled:
            logger.warning(f"  - {class_name}")
        logger.warning("!" * 60)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the mask-to-YOLO converter."""
    # Setup logging
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(LOG_FILE)
    
    logger.info("=" * 60)
    logger.info("Mask-to-YOLO Label Converter")
    logger.info("=" * 60)
    logger.info(f"Input root: {INPUT_ROOT.absolute()}")
    logger.info(f"Output root: {OUTPUT_ROOT.absolute()}")
    logger.info(f"Color tolerance: {COLOR_TOLERANCE}")
    logger.info(f"Split ratio: train={SPLIT_RATIO['train']}, valid={SPLIT_RATIO['valid']}, test={SPLIT_RATIO['test']}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info("")
    
    # Validate class colors
    classes_without_color = []
    for class_id, class_info in CLASS_COLORS.items():
        if class_info["enabled"] and class_info["color"] is None:
            classes_without_color.append(class_info["name"])
    
    if classes_without_color:
        logger.warning(f"{len(classes_without_color)} enabled classes have no color set:")
        for name in classes_without_color:
            logger.warning(f"  - {name}")
        logger.warning("These classes will be skipped during processing.")
        logger.info("")
    
    # Process all mask frames
    logger.info("Processing mask frames...")
    class_counts = process_video_dataset(INPUT_ROOT, OUTPUT_ROOT, logger)
    logger.info("")
    
    # Print class summary
    print_class_summary(class_counts, CLASS_COLORS, logger)
    logger.info("")
    
    # Split dataset
    logger.info("Splitting dataset into train/valid/test...")
    split_counts = split_dataset(INPUT_ROOT, OUTPUT_ROOT, SPLIT_RATIO, RANDOM_SEED, logger)
    logger.info("")

    # Write data.yaml
    generate_data_yaml(OUTPUT_ROOT, CLASS_COLORS, logger)
    logger.info("")

    # Final summary
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {sum(split_counts.values())}")
    logger.info(f"  Train: {split_counts.get('train', 0)}")
    logger.info(f"  Valid: {split_counts.get('valid', 0)}")
    logger.info(f"  Test:  {split_counts.get('test', 0)}")
    logger.info(f"Total instances: {sum(class_counts.values())}")
    logger.info(f"Log file: {LOG_FILE.absolute()}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
