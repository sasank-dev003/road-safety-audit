"""
generate_ae_review_list.py
---------------------------
Builds a single, AE-ready review list from a verify_labels.py run:

  1. Flagged frames (from flagged_frames.txt) — one stem per line, in priority order.
  2. For every class with ZERO instances found, a candidate list of frames that
     contain its known color-neighbor classes — the frames most likely to hold
     a mis-picked swatch for the missing class, instead of a genuinely absent one.

Run this AFTER verify_labels.py, from the same video folder (it reuses
verify_output/flagged_frames.txt and reads labels/ directly).

Output: verify_output/ae_review_list.txt — copy stems from here straight into
After Effects' project search, one at a time.
"""

import os
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION — match verify_labels.py for this run
# =============================================================================

LABELS_DIR = "labels"
OUTPUT_DIR = "verify_output"
FLAGGED_FRAMES_FILE = "flagged_frames.txt"   # produced by verify_labels.py

# Same class order/ids as mask_to_yolo.py / verify_labels.py — class_id in the
# label files is a plain index into this list.
EXPECTED_CLASSES = [
    "SHOULDER_LANE", "CENTRE_LANE", "KERB", "RIGID_CRASH_BARRIER",
    "W-BEAM_CRASH_BARRIER", "CHEVRON_SIGNS", "MANDATORY_SIGNS",
    "WARNING_SIGNS", "INFORMATIVE_SIGNS",
]

# For each class, which OTHER classes are close enough in AE color that a
# mis-pick could plausibly have landed there instead. Used only when a class
# comes back with 0 instances — narrows "check the whole video" down to
# "check frames that already contain one of these".
# Classes not listed here (or with an empty list) have no color-confusable
# neighbor — a 0-instance result for them just needs a manual footage check
# (does this stretch of road even have one?), not a frame-level color audit.
COLOR_NEIGHBOR_MAP = {
    "INFORMATIVE_SIGNS": ["WARNING_SIGNS", "CHEVRON_SIGNS", "CENTRE_LANE"],
    "CHEVRON_SIGNS":      ["WARNING_SIGNS", "CENTRE_LANE", "INFORMATIVE_SIGNS"],
    "WARNING_SIGNS":      ["CHEVRON_SIGNS", "CENTRE_LANE", "INFORMATIVE_SIGNS"],
    "CENTRE_LANE":        ["CHEVRON_SIGNS", "WARNING_SIGNS", "INFORMATIVE_SIGNS"],
    "KERB":               [],   # light green — no real neighbor, absence is more likely genuine
    "RIGID_CRASH_BARRIER": [],  # dark green — no real neighbor, absence is more likely genuine
}

# Cap how many candidate frames to list per missing class (there can be a lot —
# this is meant to be a quick spot-check list, not every single frame)
MAX_CANDIDATES_PER_CLASS = 15


def parse_label_file(path: Path):
    """Return the set of class names present in one label file."""
    present = set()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cid = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                if 0 <= cid < len(EXPECTED_CLASSES):
                    present.add(EXPECTED_CLASSES[cid])
    except Exception:
        pass
    return present


def main():
    base = Path.cwd()
    labels_dir = base / LABELS_DIR
    output_dir = base / OUTPUT_DIR
    flagged_path = output_dir / FLAGGED_FRAMES_FILE

    if not labels_dir.exists():
        print(f"ERROR: labels dir not found: {labels_dir}")
        return

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"ERROR: no label files found in {labels_dir}")
        return

    # ── Scan every label file once: total per-class counts + per-frame class sets ──
    class_totals = defaultdict(int)
    frame_classes = {}   # stem -> set of class names present in that frame
    for lf in label_files:
        present = parse_label_file(lf)
        frame_classes[lf.stem] = present
        for c in present:
            class_totals[c] += 1

    zero_classes = [c for c in EXPECTED_CLASSES if class_totals.get(c, 0) == 0]

    # ── Flagged frames (from the verify_labels.py run) ──
    flagged_stems = []
    if flagged_path.exists():
        with open(flagged_path, "r") as f:
            flagged_stems = [ln.strip() for ln in f if ln.strip()]
    else:
        print(f"NOTE: {flagged_path} not found — run verify_labels.py first for this section. Continuing without it.")

    # ── Build the review list ──
    lines = []
    lines.append("=" * 70)
    lines.append("AE REVIEW LIST")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"SECTION 1 — FLAGGED FRAMES ({len(flagged_stems)})")
    lines.append("Copy-paste each stem into AE project search, in this order.")
    lines.append("-" * 70)
    if flagged_stems:
        lines.extend(flagged_stems)
    else:
        lines.append("(none — run verify_labels.py to populate this section)")
    lines.append("")

    lines.append(f"SECTION 2 — ZERO-INSTANCE CLASS CHECK ({len(zero_classes)} class(es))")
    lines.append("-" * 70)
    if not zero_classes:
        lines.append("All classes have at least one instance — nothing to check here.")
    for cname in zero_classes:
        neighbors = COLOR_NEIGHBOR_MAP.get(cname, [])
        lines.append("")
        lines.append(f"[{cname}] — 0 instances found")
        if not neighbors:
            lines.append("  No color-confusable neighbor for this class.")
            lines.append("  Action: manually confirm this stretch of footage genuinely has no")
            lines.append(f"  {cname.replace('_', ' ').lower()} visible anywhere before assuming it's a color-mapping bug.")
        else:
            candidate_stems = sorted(
                stem for stem, classes in frame_classes.items()
                if classes & set(neighbors)
            )
            lines.append(f"  Color-confusable with: {', '.join(neighbors)}")
            lines.append(f"  {len(candidate_stems)} frame(s) contain a neighbor class — "
                          f"showing up to {MAX_CANDIDATES_PER_CLASS}:")
            for stem in candidate_stems[:MAX_CANDIDATES_PER_CLASS]:
                lines.append(f"    {stem}")
            if len(candidate_stems) > MAX_CANDIDATES_PER_CLASS:
                lines.append(f"    ... and {len(candidate_stems) - MAX_CANDIDATES_PER_CLASS} more")

    lines.append("")
    lines.append("=" * 70)

    out_path = output_dir / "ae_review_list.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written: {out_path}")
    print(f"  Flagged frames   : {len(flagged_stems)}")
    print(f"  Zero-instance    : {len(zero_classes)} class(es) -> {zero_classes}")


if __name__ == "__main__":
    main()
