# MESADEnv — HistoGym-style RL Environment for MESAD

Adapts [HistoGym](https://github.com/XjtuAI/HistoGym)'s spatial-navigation
paradigm to work with the **MESAD** dataset (JPG image patches + TSV
annotations) instead of whole-slide images.

---

## How it works

HistoGym teaches an agent to navigate whole-slide images by moving a cursor
(UP / DOWN / LEFT / RIGHT) and zooming in or out.  MESADEnv applies the same
idea to your MESAD patches:

```
┌───────┬───────┬───────┬───────┐
│  0    │  1    │  2    │  3    │
│       │       │       │       │
├───────┼───────┼───────┼───────┤
│  4    │  5  ◄─┤  6    │  7    │  ← cursor at cell 5
│       │  ★    │       │       │
├───────┼───────┼───────┼───────┤
│  8    │  9    │ 10    │ 11    │
│       │       │       │       │
├───────┼───────┼───────┼───────┤
│ 12    │ 13    │ 14    │ 15    │
│       │       │       │       │
└───────┴───────┴───────┴───────┘
  4 × 4 grid = 16 possible biopsy sites
```

Each image is divided into a **4 × 4 grid** of 16 cells.  
The agent navigates the grid and then **SELECT**s where to biopsy.  
Reward is based on whether the selected cell overlaps an annotated lesion.

| Action | Code | Effect |
|--------|------|--------|
| UP     | 0    | Move cursor up one row |
| DOWN   | 1    | Move cursor down one row |
| LEFT   | 2    | Move cursor left one column |
| RIGHT  | 3    | Move cursor right one column |
| SELECT | 4    | Commit current cell as biopsy site |

| Outcome | Reward |
|---------|--------|
| SELECT hits an annotated region (IoU ≥ 0.3) | **+10** |
| Navigation step | **−0.05** |
| SELECT misses all annotations | **−2** |
| Image has no annotations | **0** |
| Episode exceeds max_steps (timeout) | **−2** |

---

## File structure

```
mesad_histogym/
├── mesad_env.py              ← main HistoGym-style environment
├── train_mesad.py            ← training script (DQN via stable-baselines3)
├── evaluate_mesad.py         ← evaluation + metrics (Acc/Prec/Rec/F1)
├── quick_test.py             ← sanity check without training
├── utils/
│   └── mesad_annotation.py  ← TSV annotation parser + grid utilities
└── README.md
```

---

## Requirements

```bash
pip install stable-baselines3 gymnasium opencv-python scikit-learn
```

---

## Dataset layout expected

```
mesad-real/
├── train/
│   ├── images/       ← *.jpg  or *.png
│   └── annotations/  ← *.tsv  (one file per image, same base-name)
└── val/
    ├── images/
    └── annotations/
```

### TSV annotation format (auto-detected)

The parser handles the most common variants automatically:

**Variant A — bounding box (x1 y1 x2 y2):**
```
image_name	label	x1	y1	x2	y2
patch_001.jpg	lesion	120	80	240	180
```

**Variant B — xywh:**
```
image_name	label	x	y	width	height
patch_001.jpg	lesion	120	80	120	100
```

**Variant C — label only (no spatial info):**
```
image_name	label
patch_001.jpg	lesion
```
> In variant C the environment gives a neutral reward (0) on SELECT because
> there is no spatial target — the image-level label is preserved in `info`
> but not used for reward shaping by default.

---

## Quick start

### 1 — Sanity check

```bash
cd mesad_histogym
python quick_test.py \
    --train_images ../mesad-real/train/images \
    --train_annots ../mesad-real/train/annotations
```

Expected output: SB3 env check **PASSED**, random rollout stats printed.

### 2 — Train

```bash
python train_mesad.py \
    --train_images ../mesad-real/train/images \
    --train_annots ../mesad-real/train/annotations \
    --val_images   ../mesad-real/val/images \
    --val_annots   ../mesad-real/val/annotations \
    --timesteps    50000 \
    --save_path    models/dqn_mesad
```

A TensorBoard log is written to `./tensorboard_logs/` by default:

```bash
tensorboard --logdir ./tensorboard_logs/
```

### 3 — Evaluate

```bash
python evaluate_mesad.py \
    --model_path  models/dqn_mesad \
    --val_images  ../mesad-real/val/images \
    --val_annots  ../mesad-real/val/annotations \
    --n_episodes  200
```

Output:
```
==================================================
  MESAD Evaluation Results
==================================================
  Episodes          : 200
  Mean reward       : 4.123 ± 3.45
  Hit-rate          : 61.5%  (SELECT lands on annotation)
  Mean steps        : 7.3
--------------------------------------------------
  Accuracy          : 0.6150
  Precision         : 0.6327
  Recall            : 0.5980
  F1 Score          : 0.6149
==================================================
```

---

## Comparison to original BiopsyEnv

| Feature | Original `BiopsyEnv` | `MESADEnv` (HistoGym-style) |
|---------|---------------------|------------------------------|
| Observation | 128×128 grayscale | 128×128 **RGB** tile under cursor |
| Actions | 16 (flat site pick) | 5 (navigate + SELECT) |
| Episodes | 1 step always | Up to `max_steps` (navigation) |
| Annotations | Random `lesion_zone` | **Real TSV annotations** |
| Reward signal | ±constant | Shaped (step cost + IoU-based) |
| Compatibility | gym (old) | **gymnasium** ≥ 0.26 |

---

## Tuning tips

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--grid_cols / --grid_rows` | 4 × 4 | Increase for finer localization; decreases random hit chance |
| `--max_steps` | 20 | More steps → richer navigation; set low (5–8) to match original 1-step env |
| `--iou_thresh` | 0.3 | Lower = more lenient hits; raise for precise localization |
| `--timesteps` | 50 000 | ≥ 100 000 recommended for convergence |

To reproduce the original notebook's 1-step behavior exactly,
set `--max_steps 1` and `--grid_cols 4 --grid_rows 4` (16 cells = 16 sites).
