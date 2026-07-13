"""
visualize.py — Show the agent's predicted biopsy cell overlaid on each image.

For every image the agent runs one step and selects a grid cell.
OpenCV draws:
  • Blue grid lines dividing the image into cells
  • Green boxes  — ground-truth annotated lesion regions (from .bboxes.tsv)
  • Green fill   — selected cell background when it is a HIT
  • Red fill     — selected cell background when it is a MISS
  • Thick border matching the fill colour on the selected cell
  • Label text   — "HIT" / "MISS" / "?" in the selected cell

Usage
-----
    # Browse images one by one (press any key to advance, ESC to quit):
    python visualize.py --model_path dqn_biopsy_model \\
                        --image_dir  mesad-real/val/images \\
                        --annot_dir  mesad-real/val/annotations

    # Save all frames to a folder instead of displaying:
    python visualize.py --model_path dqn_biopsy_model \\
                        --image_dir  mesad-real/val/images \\
                        --annot_dir  mesad-real/val/annotations \\
                        --save_dir   output/viz

    # Run on a single image:
    python visualize.py --model_path dqn_biopsy_model \\
                        --image      mesad-real/val/images/real3_frame_5.jpg \\
                        --annot_dir  mesad-real/val/annotations
"""

import argparse
import os
import sys

import cv2
import numpy as np
from stable_baselines3 import DQN

from mesad_env import MESADEnv
from utils.mesad_annotation import parse_mesad_annotations, cell_to_bbox, iou


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

GRID_COLOR  = (180, 180, 180)   # light grey grid lines
ANNOT_COLOR = (0, 220, 0)       # bright green — ground-truth boxes
HIT_COLOR   = (0, 200, 0)       # green  — agent landed on lesion
MISS_COLOR  = (0, 0, 220)       # red    — agent missed
UNK_COLOR   = (0, 200, 220)     # yellow — no annotation to judge
ALPHA       = 0.25              # fill transparency


def draw_overlay(img_rgb: np.ndarray,
                 annotations: list[dict],
                 predicted_cell: int,
                 grid_cols: int = 4,
                 grid_rows: int  = 4,
                 iou_thresh: float = 0.3) -> np.ndarray:
    """
    Return a BGR image with grid, annotation boxes, and agent selection drawn.
    """
    img_h, img_w = img_rgb.shape[:2]
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    cw = img_w // grid_cols
    ch = img_h // grid_rows

    # ── Determine hit / miss ─────────────────────────────────────────────────
    spatial = [a for a in annotations
               if None not in (a["x1"], a["y1"], a["x2"], a["y2"])]

    cell_box = cell_to_bbox(predicted_cell, img_w, img_h, grid_cols, grid_rows)
    if spatial:
        hit = any(
            iou(cell_box, (a["x1"], a["y1"], a["x2"], a["y2"])) >= iou_thresh
            for a in spatial
        )
        sel_color = HIT_COLOR if hit else MISS_COLOR
        label_txt = "HIT" if hit else "MISS"
    else:
        hit = None
        sel_color = UNK_COLOR
        label_txt = "?"

    # ── Semi-transparent fill for selected cell ──────────────────────────────
    x1s, y1s, x2s, y2s = cell_box
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1s, y1s), (x2s, y2s), sel_color, cv2.FILLED)
    cv2.addWeighted(overlay, ALPHA, canvas, 1 - ALPHA, 0, canvas)

    # ── Ground-truth annotation boxes (bright green, thick) ──────────────────
    for a in spatial:
        cv2.rectangle(canvas,
                      (int(a["x1"]), int(a["y1"])),
                      (int(a["x2"]), int(a["y2"])),
                      ANNOT_COLOR, 3)
        # Label tag above the box
        if a.get("label"):
            tag_y = max(int(a["y1"]) - 6, 14)
            cv2.putText(canvas, a["label"],
                        (int(a["x1"]) + 2, tag_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ANNOT_COLOR, 1,
                        cv2.LINE_AA)

    # ── Grid lines ───────────────────────────────────────────────────────────
    for c in range(1, grid_cols):
        cv2.line(canvas, (c * cw, 0), (c * cw, img_h), GRID_COLOR, 1)
    for r in range(1, grid_rows):
        cv2.line(canvas, (0, r * ch), (img_w, r * ch), GRID_COLOR, 1)

    # ── Selected cell border ─────────────────────────────────────────────────
    cv2.rectangle(canvas, (x1s, y1s), (x2s, y2s), sel_color, 3)

    # ── HIT / MISS label centred in the selected cell ────────────────────────
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.9
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(label_txt, font, font_scale, thickness)
    cx = x1s + (x2s - x1s - tw) // 2
    cy = y1s + (y2s - y1s + th) // 2
    # Shadow for readability
    cv2.putText(canvas, label_txt, (cx + 1, cy + 1),
                font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, label_txt, (cx, cy),
                font, font_scale, sel_color, thickness, cv2.LINE_AA)

    # ── Cell index in every cell (small, top-left corner) ────────────────────
    for idx in range(grid_cols * grid_rows):
        r, c = divmod(idx, grid_cols)
        cv2.putText(canvas, str(idx),
                    (c * cw + 4, r * ch + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRID_COLOR, 1, cv2.LINE_AA)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise the agent's biopsy-site predictions with OpenCV."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image_dir", help="Directory of images to predict on")
    g.add_argument("--image",     help="Single image file")

    p.add_argument("--model_path", required=True)
    p.add_argument("--annot_dir",  default=None,
                   help="TSV annotation dir (enables HIT/MISS colouring)")
    p.add_argument("--save_dir",   default=None,
                   help="If set, save frames here instead of displaying")
    p.add_argument("--grid_cols",  type=int,   default=4)
    p.add_argument("--grid_rows",  type=int,   default=4)
    p.add_argument("--obs_size",   type=int,   default=128)
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--scale",      type=float, default=1.0,
                   help="Display scale factor (e.g. 0.5 to halve window size)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted(
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in img_exts
        )
    if not image_paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── Load model via dummy env ─────────────────────────────────────────────
    img_dir_for_env = (
        args.image_dir if args.image_dir
        else os.path.dirname(os.path.abspath(args.image))
    )
    annot_dir_for_env = args.annot_dir or img_dir_for_env
    env = MESADEnv(
        image_dir      = img_dir_for_env,
        annotation_dir = annot_dir_for_env,
        grid_cols      = args.grid_cols,
        grid_rows      = args.grid_rows,
        obs_w          = args.obs_size,
        obs_h          = args.obs_size,
        iou_thresh     = args.iou_thresh,
    )
    model = DQN.load(args.model_path, env=env)
    print(f"Model loaded: {args.model_path}")
    print(f"Processing {len(image_paths)} image(s) …")
    if not args.save_dir:
        print("Press any key to advance, ESC to quit.\n")

    hits = misses = 0

    for img_path in image_paths:
        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [skip] Cannot read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Build observation
        resized = cv2.resize(img_rgb, (args.obs_size, args.obs_size))
        gray    = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        obs     = gray[:, :, np.newaxis]

        # Predict
        action, _ = model.predict(obs, deterministic=True)
        predicted_cell = int(action)
        predicted_row, predicted_col = divmod(predicted_cell, args.grid_cols)

        # Load annotations
        annotations = []
        if args.annot_dir:
            base      = os.path.splitext(os.path.basename(img_path))[0]
            base_path = os.path.join(args.annot_dir, base)
            annotations = parse_mesad_annotations(base_path)

        # Draw
        frame = draw_overlay(
            img_rgb, annotations, predicted_cell,
            grid_cols  = args.grid_cols,
            grid_rows  = args.grid_rows,
            iou_thresh = args.iou_thresh,
        )

        # Determine result for console output
        spatial = [a for a in annotations
                   if None not in (a["x1"], a["y1"], a["x2"], a["y2"])]
        cell_box = cell_to_bbox(predicted_cell, img_w, img_h,
                                args.grid_cols, args.grid_rows)
        if spatial:
            hit = any(
                iou(cell_box, (a["x1"], a["y1"], a["x2"], a["y2"])) >= args.iou_thresh
                for a in spatial
            )
            hits   += hit
            misses += not hit
            result = "HIT ✓" if hit else "MISS ✗"
        else:
            result = "N/A"

        print(f"  {os.path.basename(img_path):35s}  "
              f"cell={predicted_cell:2d} (col={predicted_col}, row={predicted_row})  {result}")

        if args.scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(img_w * args.scale), int(img_h * args.scale)),
                interpolation=cv2.INTER_AREA,
            )

        if args.save_dir:
            out_name = os.path.splitext(os.path.basename(img_path))[0] + "_pred.jpg"
            out_path = os.path.join(args.save_dir, out_name)
            cv2.imwrite(out_path, frame)
        else:
            cv2.imshow("MESAD — Agent Prediction", frame)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:   # ESC
                break

    cv2.destroyAllWindows()
    env.close()

    # Summary
    total_scored = hits + misses
    print(f"\n{'─' * 55}")
    print(f"Images processed : {len(image_paths)}")
    if total_scored:
        print(f"Hit-rate         : {hits}/{total_scored} = {hits/total_scored*100:.1f}%")
    if args.save_dir:
        print(f"Frames saved to  : {args.save_dir}")


if __name__ == "__main__":
    main()
