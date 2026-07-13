"""
predict.py — Run a trained MESADEnv model on new images without needing
             the full dataset or annotation files.

This is the script you run locally (or anywhere) after downloading the
trained model from Google Colab.

The model is a single-step policy: it receives the full image (resized to
obs_size × obs_size grayscale) and outputs one of grid_cols × grid_rows
cell indices as the predicted biopsy site.

Usage
-----
    python predict.py \\
        --model_path  dqn_biopsy_model \\
        --image_dir   mesad-real/val/images

    # Or a single image:
    python predict.py \\
        --model_path  dqn_biopsy_model \\
        --image       mesad-real/val/images/patch_001.jpg

    # With annotation dir to also score accuracy:
    python predict.py \\
        --model_path  dqn_biopsy_model \\
        --image_dir   mesad-real/val/images \\
        --annot_dir   mesad-real/val/annotations

Output
------
    For each image: predicted grid cell index + (col, row) position.
    If --annot_dir is given: also prints hit/miss vs TSV annotations.
    Summary metrics at the end.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from stable_baselines3 import DQN

from mesad_env import MESADEnv
from utils.mesad_annotation import build_index, cell_to_bbox, iou


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a pre-trained MESAD model to predict biopsy sites."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image_dir", help="Directory of images to predict on")
    g.add_argument("--image",     help="Single image file to predict on")

    p.add_argument("--model_path",  required=True,
                   help="Path to saved model (.zip) — omit the .zip extension")
    p.add_argument("--annot_dir",   default=None,
                   help="(Optional) TSV annotation dir — enables accuracy scoring")
    p.add_argument("--grid_cols",   type=int, default=4)
    p.add_argument("--grid_rows",   type=int, default=4)
    p.add_argument("--obs_size",    type=int, default=128)
    p.add_argument("--iou_thresh",  type=float, default=0.3)
    p.add_argument("--render",      action="store_true",
                   help="Show OpenCV window with grid overlay for each image")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Single-image prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_one(model, image_path: str, annot_records: list,
                grid_cols=4, grid_rows=4,
                obs_w=128, obs_h=128,
                iou_thresh=0.3, render=False) -> dict:
    """
    Run the model on a single image (single-step episode).

    Returns a dict:
        image_path, predicted_cell, predicted_col, predicted_row,
        hit (bool or None), episode_reward
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise IOError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # Build observation: full image → grayscale → (obs_h, obs_w, 1)
    resized = cv2.resize(img_rgb, (obs_w, obs_h))
    gray    = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    obs     = gray[:, :, np.newaxis]   # (H, W, 1)

    # Model predicts a cell index directly
    action, _ = model.predict(obs, deterministic=True)
    predicted_cell = int(action)
    predicted_row, predicted_col = divmod(predicted_cell, grid_cols)

    # Score against annotations (if any)
    spatial = [a for a in annot_records
               if None not in (a["x1"], a["y1"], a["x2"], a["y2"])]
    hit = None
    ep_reward = 0.0
    if spatial:
        cell_box = cell_to_bbox(predicted_cell, img_w, img_h, grid_cols, grid_rows)
        hit = any(
            iou(cell_box, (a["x1"], a["y1"], a["x2"], a["y2"])) >= iou_thresh
            for a in spatial
        )
        ep_reward = 10.0 if hit else -2.0

    # Optional render
    if render:
        cw = img_w // grid_cols
        ch = img_h // grid_rows
        canvas = img_rgb.copy()
        for a in spatial:
            cv2.rectangle(canvas,
                          (int(a["x1"]), int(a["y1"])),
                          (int(a["x2"]), int(a["y2"])),
                          (0, 200, 0), 2)
        for c in range(1, grid_cols):
            cv2.line(canvas, (c * cw, 0), (c * cw, img_h), (100, 180, 255), 1)
        for r in range(1, grid_rows):
            cv2.line(canvas, (0, r * ch), (img_w, r * ch), (100, 180, 255), 1)
        x1, y1, x2, y2 = cell_to_bbox(predicted_cell, img_w, img_h, grid_cols, grid_rows)
        color = (0, 255, 0) if hit else (0, 0, 255) if hit is False else (0, 255, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imshow(os.path.basename(image_path), bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "image_path":     image_path,
        "predicted_cell": predicted_cell,
        "predicted_col":  predicted_col,
        "predicted_row":  predicted_row,
        "hit":            hit,
        "episode_reward": ep_reward,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Collect image paths
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

    # Build annotation index (optional)
    annot_map: dict[str, list] = {}
    if args.annot_dir:
        from utils.mesad_annotation import parse_mesad_annotations
        for img_path in image_paths:
            base      = os.path.splitext(os.path.basename(img_path))[0]
            base_path = os.path.join(args.annot_dir, base)
            annot_map[img_path] = parse_mesad_annotations(base_path)

    # Load model — needs a dummy env to validate spaces at load time
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
    print(f"Model loaded from: {args.model_path}\n")

    # Predict
    results = []
    for img_path in image_paths:
        annots = annot_map.get(img_path, [])
        res    = predict_one(
            model, img_path, annots,
            grid_cols  = args.grid_cols,
            grid_rows  = args.grid_rows,
            obs_w      = args.obs_size,
            obs_h      = args.obs_size,
            iou_thresh = args.iou_thresh,
            render     = args.render,
        )
        results.append(res)

        hit_str = {True: "HIT ✓", False: "MISS ✗", None: "N/A"}[res["hit"]]
        print(f"{os.path.basename(img_path):30s}  "
              f"cell={res['predicted_cell']:2d}  "
              f"(col={res['predicted_col']}, row={res['predicted_row']})  "
              f"reward={res['episode_reward']:5.1f}  {hit_str}")

    # Summary
    if results:
        scored = [r for r in results if r["hit"] is not None]
        print(f"\n{'─' * 60}")
        print(f"Images processed : {len(results)}")
        print(f"Mean reward      : {np.mean([r['episode_reward'] for r in results]):.3f}")
        if scored:
            hits = sum(1 for r in scored if r["hit"])
            print(f"Hit-rate         : {hits}/{len(scored)} = {hits/len(scored)*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
