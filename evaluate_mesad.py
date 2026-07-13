"""
evaluate_mesad.py — Evaluate a trained DQN agent on the MESAD dataset.

Usage
-----
    python evaluate_mesad.py \
        --model_path  models/dqn_mesad \
        --val_images  mesad-real/val/images \
        --val_annots  mesad-real/val/annotations \
        --n_episodes  200

Reports:
    • Mean / std episode reward
    • Hit-rate    (% of SELECT actions landing on an annotated region)
    • Mean steps  to SELECT
    • Accuracy / Precision / Recall / F1  (binary: hit vs miss)
"""

import argparse

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from stable_baselines3 import DQN

from mesad_env import MESADEnv


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained DQN model on MESAD validation data."
    )
    p.add_argument("--model_path",  required=True,
                   help="Path to saved model (.zip) — omit the .zip extension")
    p.add_argument("--val_images",  required=True,
                   help="Path to validation images directory")
    p.add_argument("--val_annots",  required=True,
                   help="Path to validation annotations (TSV) directory")
    p.add_argument("--n_episodes",  type=int, default=200,
                   help="Number of episodes to evaluate")
    p.add_argument("--grid_cols",   type=int, default=4)
    p.add_argument("--grid_rows",   type=int, default=4)
    p.add_argument("--obs_size",    type=int, default=128)
    p.add_argument("--render",      action="store_true",
                   help="Show OpenCV render window during evaluation")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic policy (default: True)")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, env, n_episodes: int, deterministic: bool, render: bool):
    """
    Run n_episodes episodes and collect statistics.

    Returns
    -------
    dict with keys:
        rewards, steps_to_select, hits, y_true, y_pred
    """
    rewards         = []
    steps_to_select = []
    hits            = []  # 1 = hit annotated region, 0 = miss or no annotation
    y_true          = []  # ground-truth: 1 if image has annotations, else 0
    y_pred          = []  # predicted:    1 if agent hit an annotation, else 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        step      = 0
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            step      += 1
            done       = terminated or truncated

            if render:
                env.render(mode="human")

        rewards.append(ep_reward)
        steps_to_select.append(step)

        # Determine hit/miss from final reward
        # +10 → hit, -2 → miss, 0 → unannotated image
        has_annotations = info["n_annotations"] > 0
        hit             = ep_reward >= 10.0

        y_true.append(1 if has_annotations else 0)
        y_pred.append(1 if hit             else 0)
        hits.append(hit)

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}  "
                  f"reward={ep_reward:.2f}  steps={step}")

    return {
        "rewards":         rewards,
        "steps_to_select": steps_to_select,
        "hits":            hits,
        "y_true":          y_true,
        "y_pred":          y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Build environment ──────────────────────────────────────────────────
    env = MESADEnv(
        image_dir      = args.val_images,
        annotation_dir = args.val_annots,
        grid_cols      = args.grid_cols,
        grid_rows      = args.grid_rows,
        obs_w          = args.obs_size,
        obs_h          = args.obs_size,
        seed           = args.seed,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    model = DQN.load(args.model_path, env=env)
    print(f"Loaded model from: {args.model_path}")
    print(f"Evaluating over {args.n_episodes} episodes …\n")

    # ── Run ────────────────────────────────────────────────────────────────
    results = run_evaluation(
        model,
        env,
        n_episodes   = args.n_episodes,
        deterministic = args.deterministic,
        render       = args.render,
    )

    # ── Metrics ────────────────────────────────────────────────────────────
    rewards         = results["rewards"]
    steps           = results["steps_to_select"]
    hits            = results["hits"]
    y_true          = results["y_true"]
    y_pred          = results["y_pred"]

    mean_reward = np.mean(rewards)
    std_reward  = np.std(rewards)
    hit_rate    = np.mean(hits) * 100
    mean_steps  = np.mean(steps)

    # sklearn metrics (ignore episodes with no annotations for accuracy)
    acc  = accuracy_score(y_true,  y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,    y_pred, zero_division=0)
    f1   = f1_score(y_true,        y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("  MESAD Evaluation Results")
    print("=" * 50)
    print(f"  Episodes          : {args.n_episodes}")
    print(f"  Mean reward       : {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"  Hit-rate          : {hit_rate:.1f}%  (SELECT lands on annotation)")
    print(f"  Mean steps        : {mean_steps:.2f}")
    print("-" * 50)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print("=" * 50)
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, target_names=["no annotation", "annotation hit"]))

    env.close()


if __name__ == "__main__":
    main()
