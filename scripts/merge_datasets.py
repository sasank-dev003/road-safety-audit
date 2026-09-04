"""
merge_datasets.py
-----------------
Merges multiple video datasets into one combined dataset,
then re-splits into train/val/test.

Folder structure expected per video:
    ROAD_VIDEO_X/
        original_frames/   <-- original road frames (.png)
        labels/            <-- YOLO .txt label files

Output:
    merged/
        images/            <-- all frames combined
        labels/            <-- all labels combined
        split/
            train/images, train/labels
            val/images,   val/labels
            test/images,  test/labels
"""

import os
import shutil
import random
import logging

# ─────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────

# Add all your video dataset roots here
VIDEO_DATASETS = [
    r"C:\Users\sasi3\Desktop\project\AfterFX\ROAD_VIDEO_1",
    r"C:\Users\sasi3\Desktop\project\AfterFX\ROAD_VIDEO_2",
    r"C:\Users\sasi3\Desktop\project\AfterFX\ROAD_VIDEO_3",
    # add more as needed...
]

# Subfolder names inside each video dataset
IMAGES_SUBDIR = "original_frames"
LABELS_SUBDIR = "labels"

# Where to save the merged + split output
MERGED_DIR = r"C:\Users\sasi3\Desktop\project\AfterFX\merged"

# File extension for images
IMAGE_EXT = ".png"

# Train / Val / Test split ratio
SPLIT_RATIO = (0.8, 0.1, 0.1)

# Random seed — same seed = same split every run
SPLIT_SEED = 42

# ─────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────

os.makedirs(MERGED_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(MERGED_DIR, "merge_log.txt"))
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────

def merge_datasets(video_datasets, merged_dir, images_subdir, labels_subdir):
    """
    Copies all images and labels from each video dataset into one merged folder.
    Prefixes filenames with video index to avoid name collisions.
    e.g. frame_00001.png from video 1 -> v1_frame_00001.png
    """
    merged_images = os.path.join(merged_dir, "images")
    merged_labels = os.path.join(merged_dir, "labels")
    os.makedirs(merged_images, exist_ok=True)
    os.makedirs(merged_labels, exist_ok=True)

    total_images = 0
    total_labels = 0

    for idx, video_root in enumerate(video_datasets):
        prefix = f"v{idx+1}_"
        images_dir = os.path.join(video_root, images_subdir)
        labels_dir = os.path.join(video_root, labels_subdir)

        if not os.path.exists(images_dir):
            log.warning(f"Images folder not found, skipping: {images_dir}")
            continue
        if not os.path.exists(labels_dir):
            log.warning(f"Labels folder not found, skipping: {labels_dir}")
            continue

        log.info(f"[VIDEO {idx+1}] {video_root}")

        # Copy images
        img_count = 0
        for fname in os.listdir(images_dir):
            if fname.endswith(IMAGE_EXT):
                src = os.path.join(images_dir, fname)
                dst = os.path.join(merged_images, prefix + fname)
                shutil.copy2(src, dst)
                img_count += 1

        # Copy labels
        lbl_count = 0
        for fname in os.listdir(labels_dir):
            if fname.endswith(".txt"):
                src = os.path.join(labels_dir, fname)
                dst = os.path.join(merged_labels, prefix + fname)
                shutil.copy2(src, dst)
                lbl_count += 1

        log.info(f"  Copied {img_count} images, {lbl_count} labels")
        total_images += img_count
        total_labels += lbl_count

    log.info(f"")
    log.info(f"Total merged: {total_images} images, {total_labels} labels")
    return merged_images, merged_labels


# ─────────────────────────────────────────
# SPLIT
# ─────────────────────────────────────────

def split_dataset(merged_images, merged_labels, merged_dir, split=SPLIT_RATIO, seed=SPLIT_SEED):
    """
    Splits merged dataset into train/val/test.
    Only splits stems that have both an image and a label.
    """
    random.seed(seed)

    # Get stems with both image and label
    image_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(merged_images)
        if f.endswith(IMAGE_EXT)
    }
    label_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(merged_labels)
        if f.endswith(".txt")
    }

    matched = sorted(image_stems & label_stems)
    unmatched = (image_stems | label_stems) - (image_stems & label_stems)

    if unmatched:
        log.warning(f"{len(unmatched)} files have no matching pair — skipping them")

    random.shuffle(matched)

    total     = len(matched)
    train_end = int(total * split[0])
    val_end   = train_end + int(total * split[1])

    splits = {
        "train": matched[:train_end],
        "val":   matched[train_end:val_end],
        "test":  matched[val_end:]
    }

    split_dir = os.path.join(merged_dir, "split")

    for split_name, stems in splits.items():
        for folder in ["images", "labels"]:
            os.makedirs(os.path.join(split_dir, split_name, folder), exist_ok=True)

        for stem in stems:
            shutil.copy2(
                os.path.join(merged_images, stem + IMAGE_EXT),
                os.path.join(split_dir, split_name, "images", stem + IMAGE_EXT)
            )
            shutil.copy2(
                os.path.join(merged_labels, stem + ".txt"),
                os.path.join(split_dir, split_name, "labels", stem + ".txt")
            )

        log.info(f"  {split_name:5s}: {len(stems)} frames")

    log.info(f"Split saved to: {split_dir}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("STEP 1: Merging datasets")
    log.info("=" * 50)

    merged_images, merged_labels = merge_datasets(
        VIDEO_DATASETS, MERGED_DIR, IMAGES_SUBDIR, LABELS_SUBDIR
    )

    log.info("")
    log.info("=" * 50)
    log.info("STEP 2: Splitting into train / val / test")
    log.info("=" * 50)
    log.info(f"  Ratio -- train:{SPLIT_RATIO[0]}  val:{SPLIT_RATIO[1]}  test:{SPLIT_RATIO[2]}")
    log.info(f"  Seed  -- {SPLIT_SEED}")

    split_dataset(merged_images, merged_labels, MERGED_DIR)

    log.info("")
    log.info("=" * 50)
    log.info("DONE")
    log.info("=" * 50)
    log.info(f"Merged dataset: {MERGED_DIR}")
    log.info(f"Update your data.yaml path to: {os.path.join(MERGED_DIR, 'split')}")


if __name__ == "__main__":
    main()
