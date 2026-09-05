# Road Safety Audit — Computer Vision Pipeline

Automated road infrastructure detection from dashcam footage, built as part of the
**AI Automated Road Safety Audit** project. A YOLO11 segmentation model is
trained to detect the same road elements a human safety auditor would look
for — lane markings, kerbs, crash barriers, and road signage — directly from
ordinary dashcam video.

This repo covers the pipeline **through 9-class detection**: data annotation →
label conversion → verification → training → inference. Compliance-checking
modules (e.g. sign mounting-height auditing against IRC:67-2012) are a
separate, later phase and not included here.

## The 9 classes

**Segmentation** (pixel-level masks — shape matters):
- `SHOULDER_LANE`
- `CENTRE_LANE`
- `KERB`
- `RIGID_CRASH_BARRIER`
- `W-BEAM_CRASH_BARRIER`

**Detection** (bounding boxes — only location/extent matters):
- `CHEVRON_SIGNS`
- `MANDATORY_SIGNS`
- `WARNING_SIGNS`
- `INFORMATIVE_SIGNS`

## Pipeline overview

```
raw dashcam video
       │
       ▼
frame extraction + hand-rotoscoping in Adobe After Effects
(each class painted as a flat, unique solid color → "mask frames")
       │
       ▼
mask_to_yolo.py          → converts color-coded masks into YOLO segmentation
                            labels (.txt), splits into train/valid/test
       │
       ▼
verify_labels.py         → renders labels back onto frames + flags anomalous
                            frames (bad polygons, class outliers, overlaps)
generate_ae_review_list.py → builds a prioritized list of frames to re-check
                              in After Effects, from a verify_labels.py run
       │
       ▼
merge_datasets.py        → combines multiple per-video datasets into one
                            unified train/val/test set
       │
       ▼
train.py                 → trains YOLO11-seg, auto-detects GPU tier
                            (batch/workers scale to VRAM), road-tuned
                            augmentation, multi-scale training
       │
       ▼
predict.py                → runs inference on new video, draws segmentation
                             masks + sign bounding boxes
```

## Scripts

| Script | Purpose |
|---|---|
| `mask_to_yolo.py` | Converts After Effects color-coded mask frames into YOLO segmentation label files. Per-class color mapping, per-class polygon tuning (epsilon/min-area), auto train/valid/test split, `data.yaml` generation. |
| `verify_labels.py` | QA tool — renders generated labels back onto frames for visual spot-checking, computes per-class stats, and flags anomalous frames (empty labels, malformed polygons, class-count outliers, `SHOULDER_LANE`/`CENTRE_LANE` overlap, etc.) *before* burning a training run on bad annotations. |
| `generate_ae_review_list.py` | Turns a `verify_labels.py` run into a single AE-ready review list — flagged frames plus candidate frames for any class that came back with zero instances (likely a color mis-pick). |
| `merge_datasets.py` | Combines multiple per-video datasets (each already converted to YOLO labels) into one merged, re-split dataset. |
| `train.py` | Trains YOLO11-seg. Auto-detects GPU and scales batch/workers to VRAM (runs unmodified on high-end or entry-level GPUs), builds the active class list dynamically, applies road-scene-tuned augmentation, evaluates the best checkpoint on the test set. |
| `predict.py` | Runs the trained model on a video: segmentation classes drawn as translucent masks (bitmap-based, alpha-blended straight from the raw predicted mask — no polygon artifacts), sign classes drawn as bounding boxes. |
| `color_config.py` | Small GUI (Tkinter) for picking/adjusting the display colors and mask opacity used in `predict.py`, and writing them back into the script. |

## Requirements

```
pip install -r requirements.txt
```

`torch`/`torchvision` aren't pinned in `requirements.txt` — install the build
that matches your CUDA version from https://pytorch.org first, then install
the rest.

## Usage

Each script has a `# USER CONFIGURATION` block at the top with paths and
settings to edit before running — no command-line arguments needed.

1. Rotoscope footage in After Effects, export original + color-coded mask
   frame sequences.
2. Set paths in `mask_to_yolo.py`, run it → get `labels/` + `split/` for one video.
3. Run `verify_labels.py` on that video's frames/labels → check `verify_output/`.
4. If issues are flagged, run `generate_ae_review_list.py` → get a prioritized
   list to fix in After Effects, then re-run step 2.
5. Repeat 1–4 per video, then run `merge_datasets.py` to combine everything.
6. Set `DATASET_ROOT` in `train.py`, run it.
7. Point `predict.py` at the trained `best.pt` and a new video.

## Status

Currently trained/validated on: `SHOULDER_LANE`, `CENTRE_LANE`, `KERB`,
`RIGID_CRASH_BARRIER`, `W-BEAM_CRASH_BARRIER`. Sign classes
(`CHEVRON_SIGNS`, `MANDATORY_SIGNS`, `WARNING_SIGNS`, `INFORMATIVE_SIGNS`)
are annotated incrementally as footage is processed — `INFORMATIVE_SIGNS` in
particular still has limited data volume.

Next phase (not in this repo): IRC:67-2012 compliance checking, starting with
monocular sign mounting-height estimation.
