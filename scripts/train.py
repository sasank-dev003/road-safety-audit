"""
YOLOv11 Segmentation Training Script
Road Safety Audit Project — IIT Bhubaneswar

Portable: auto-detects GPU and adjusts batch size / workers accordingly.
Toggle classes on/off via ACTIVE_CLASSES below.
"""

import os
import sys
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
#  USER CONFIGURATION — edit this section only
# ─────────────────────────────────────────────

# Root of your dataset (contains train/ val/ test/ folders)
DATASET_ROOT = r"C:\Users\YourName\dataset\split"

# Where to save training runs
OUTPUT_DIR = r"C:\Users\YourName\runs"

# YOLOv11 model size: n / s / m / l / x  (n=fastest, x=most accurate)
MODEL_SIZE = "s"

# Training hyperparameters
EPOCHS        = 100
IMG_SIZE      = 640
LR0           = 0.01       # initial learning rate
LRF           = 0.01       # final lr = LR0 * LRF
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3.0
IOU_THRESHOLD = 0.7        # NMS IoU threshold

# Early stopping: stop if no improvement for this many epochs (0 = disabled)
PATIENCE = 20

# WandB logging: set to True if you have wandb installed and want cloud dashboards
USE_WANDB = False
WANDB_PROJECT = "road-safety-audit"

# ─────────────────────────────────────────────
#  CLASS CONFIGURATION
#  Set enabled=True only for classes that have annotated data.
#  Script will auto-assign sequential IDs to enabled classes.
# ─────────────────────────────────────────────

ALL_CLASSES = [
    {"name": "SHOULDER_LANE",         "enabled": True},
    {"name": "CENTRE_LANE",           "enabled": True},
    {"name": "KERB",                  "enabled": True},
    {"name": "RIGID_CRASH_BARRIER",   "enabled": True},
    {"name": "W-BEAM_CRASH_BARRIER",  "enabled": True},
    {"name": "CHEVRON_SIGNS",         "enabled": False},  # no data yet
    {"name": "INFORMATIVE_SIGNS",     "enabled": False},  # no data yet
    {"name": "MANDATORY_SIGNS",       "enabled": False},  # no data yet
    {"name": "WARNING_SIGNS",         "enabled": False},  # no data yet
]

# ─────────────────────────────────────────────
#  AUTO CONFIGURATION — don't edit below unless you know what you're doing
# ─────────────────────────────────────────────

def detect_hardware():
    """Detect available GPU and return appropriate training settings."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

            print(f"[GPU] {gpu_name} — {vram_gb:.1f} GB VRAM")

            if vram_gb >= 15:           # Kaggle T4, lab GPU
                batch, workers = 16, 8
                tier = "high-end"
            elif vram_gb >= 8:          # mid-range
                batch, workers = 8, 4
                tier = "mid-range"
            else:                       # RTX 4050 6GB and similar
                batch, workers = 4, 2
                tier = "low VRAM"

            print(f"[AUTO] Tier: {tier} → batch={batch}, workers={workers}")
            return "0", batch, workers  # device string for YOLO

        else:
            print("[WARN] No CUDA GPU found — training on CPU (will be very slow)")
            return "cpu", 2, 0

    except ImportError:
        print("[ERROR] PyTorch not installed. Run: pip install torch torchvision")
        sys.exit(1)


def get_active_classes():
    active = [c["name"] for c in ALL_CLASSES if c["enabled"]]
    if not active:
        print("[ERROR] No classes are enabled. Set at least one enabled=True in ALL_CLASSES.")
        sys.exit(1)
    print(f"[CLASSES] {len(active)} active: {', '.join(active)}")
    return active


def build_data_yaml(dataset_root: Path, active_classes: list, output_dir: Path) -> Path:
    """Generate data.yaml for the active classes."""
    data = {
        "path"  : str(dataset_root),
        "train" : "train/images",
        "val"   : "val/images",
        "test"  : "test/images",
        "nc"    : len(active_classes),
        "names" : active_classes,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[YAML] Written to {yaml_path}")
    return yaml_path


def check_dataset(dataset_root: Path, active_classes: list):
    """Basic sanity checks on the dataset structure."""
    issues = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            issues.append(f"Missing: {img_dir}")
        if not lbl_dir.exists():
            issues.append(f"Missing: {lbl_dir}")
        else:
            imgs   = list(img_dir.glob("*.*")) if img_dir.exists() else []
            labels = list(lbl_dir.glob("*.txt"))
            print(f"[{split.upper():5s}] {len(imgs)} images, {len(labels)} labels")

    if issues:
        print("\n[ERROR] Dataset issues found:")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)


def install_dependencies():
    """Install required packages if missing."""
    packages = {"ultralytics": "ultralytics", "yaml": "pyyaml"}
    if USE_WANDB:
        packages["wandb"] = "wandb"

    for module, pkg in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[INSTALL] Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def train():
    install_dependencies()
    from ultralytics import YOLO

    # ── Setup ──
    active_classes = get_active_classes()
    device, batch, workers = detect_hardware()

    dataset_root = Path(DATASET_ROOT)
    run_name     = f"yolo11{MODEL_SIZE}_seg_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir   = Path(OUTPUT_DIR) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    check_dataset(dataset_root, active_classes)
    yaml_path = build_data_yaml(dataset_root, active_classes, output_dir)

    # ── WandB ──
    if USE_WANDB:
        try:
            import wandb
            wandb.init(project=WANDB_PROJECT, name=run_name)
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}. Continuing without it.")

    # ── Load model ──
    model_name = f"yolo11{MODEL_SIZE}-seg.pt"
    print(f"\n[MODEL] Loading {model_name} ...")
    model = YOLO(model_name)  # downloads pretrained weights automatically

    # ── Train ──
    print(f"\n[TRAIN] Starting — run: {run_name}")
    print(f"        Epochs={EPOCHS}, ImgSize={IMG_SIZE}, Batch={batch}, Device={device}\n")

    results = model.train(
        data        = str(yaml_path),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = batch,
        workers     = workers,
        device      = device,
        project     = str(Path(OUTPUT_DIR)),
        name        = run_name,
        exist_ok    = True,

        # Optimizer
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay= WEIGHT_DECAY,
        warmup_epochs= WARMUP_EPOCHS,

        # Early stopping & checkpointing
        patience    = PATIENCE,
        save        = True,
        save_period = 10,      # also save every N epochs (in addition to best)

        # IoU
        iou         = IOU_THRESHOLD,

        # Logging
        plots       = True,    # post-training validation plots
        val         = True,
        verbose     = True,
    )

    # ── Post-training ──
    best_weights = Path(OUTPUT_DIR) / run_name / "weights" / "best.pt"
    print(f"\n[DONE] Training complete.")
    print(f"[BEST] {best_weights}")
    print(f"[LOGS] {Path(OUTPUT_DIR) / run_name}")

    # Run validation on test set with best weights
    print("\n[TEST] Running final evaluation on test set...")
    best_model = YOLO(str(best_weights))
    test_results = best_model.val(
        data    = str(yaml_path),
        split   = "test",
        imgsz   = IMG_SIZE,
        device  = device,
        plots   = True,
        verbose = True,
    )

    print("\n[SUMMARY]")
    print(f"  mAP50      : {test_results.seg.map50:.4f}")
    print(f"  mAP50-95   : {test_results.seg.map:.4f}")
    print(f"  Run folder : {Path(OUTPUT_DIR) / run_name}")

    if USE_WANDB:
        try:
            import wandb
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    train()
