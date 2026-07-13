"""
train_mesad.py — Train a DQN agent on the MESAD dataset using MESADEnv.

Usage
-----
    python train_mesad.py \
        --train_images  mesad-real/train/images \
        --train_annots  mesad-real/train/annotations \
        --timesteps     50000 \
        --save_path     models/dqn_mesad

Optional flags
--------------
    --val_images    mesad-real/val/images       (used for periodic eval)
    --val_annots    mesad-real/val/annotations
    --grid_cols     4
    --grid_rows     4
    --obs_size      128
    --buffer_size   10000
    --batch_size    64
    --lr            0.0001
    --eval_freq     5000
    --n_eval_eps    50
    --seed          42
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from mesad_env import MESADEnv


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train DQN on MESAD dataset with a HistoGym-style environment."
    )
    p.add_argument("--train_images", required=True,
                   help="Path to training images directory")
    p.add_argument("--train_annots", required=True,
                   help="Path to training annotations (TSV) directory")
    p.add_argument("--val_images",   default=None,
                   help="Path to validation images directory (optional)")
    p.add_argument("--val_annots",   default=None,
                   help="Path to validation annotations directory (optional)")
    p.add_argument("--timesteps",    type=int,   default=50_000,
                   help="Total environment steps to train for")
    p.add_argument("--save_path",    default="models/dqn_mesad",
                   help="Where to save the final model")
    p.add_argument("--grid_cols",    type=int,   default=4)
    p.add_argument("--grid_rows",    type=int,   default=4)
    p.add_argument("--obs_size",     type=int,   default=128,
                   help="Observation tile size (square)")
    p.add_argument("--buffer_size",  type=int,   default=10_000)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--eval_freq",    type=int,   default=5_000,
                   help="Evaluate on val set every N steps (0 to disable)")
    p.add_argument("--n_eval_eps",   type=int,   default=50,
                   help="Number of episodes per evaluation run")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--check_env",    action="store_true",
                   help="Run SB3 environment sanity check before training")
    p.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Build training environment ─────────────────────────────────────────
    train_env = MESADEnv(
        image_dir      = args.train_images,
        annotation_dir = args.train_annots,
        grid_cols      = args.grid_cols,
        grid_rows      = args.grid_rows,
        obs_w          = args.obs_size,
        obs_h          = args.obs_size,
        seed           = args.seed,
    )
    train_env = Monitor(train_env)

    if args.check_env:
        print("Running environment sanity check …")
        check_env(train_env, warn=True)
        print("Environment check passed.\n")

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")
    print(f"Dataset size      : {len(train_env.index)} images\n")

    # ── Optional validation environment ───────────────────────────────────
    callbacks = []
    if args.val_images and args.val_annots and args.eval_freq > 0:
        val_env = Monitor(MESADEnv(
            image_dir      = args.val_images,
            annotation_dir = args.val_annots,
            grid_cols      = args.grid_cols,
            grid_rows      = args.grid_rows,
            obs_w          = args.obs_size,
            obs_h          = args.obs_size,
        ))
        best_model_path = os.path.join(os.path.dirname(args.save_path), "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        eval_cb = EvalCallback(
            val_env,
            best_model_save_path = best_model_path,
            log_path             = best_model_path,
            eval_freq            = args.eval_freq,
            n_eval_episodes      = args.n_eval_eps,
            deterministic        = True,
            verbose              = 1,
        )
        callbacks.append(eval_cb)

    # ── Periodic checkpoint saving ─────────────────────────────────────────
    ckpt_dir = os.path.join(os.path.dirname(args.save_path), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq  = max(args.eval_freq, 5000) if args.eval_freq > 0 else 10_000,
        save_path  = ckpt_dir,
        name_prefix = "dqn_mesad",
    )
    callbacks.append(ckpt_cb)

    # ── TensorBoard ────────────────────────────────────────────────────────
    tb_log = None if args.no_tensorboard else "./tensorboard_logs/"

    # ── DQN model ──────────────────────────────────────────────────────────
    # CnnPolicy handles image observations automatically.
    model = DQN(
        policy       = "CnnPolicy",
        env          = train_env,
        learning_rate = args.lr,
        buffer_size   = args.buffer_size,
        batch_size    = args.batch_size,
        train_freq    = 4,
        target_update_interval = 1000,
        exploration_fraction   = 0.2,
        exploration_final_eps  = 0.05,
        verbose                = 1,
        seed                   = args.seed,
        tensorboard_log        = tb_log,
    )

    print(f"Training for {args.timesteps:,} timesteps …\n")
    model.learn(
        total_timesteps = args.timesteps,
        callback        = callbacks,
    )

    # ── Save final model ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model.save(args.save_path)
    print(f"\nModel saved to: {args.save_path}.zip")

    # ── Quick final evaluation ─────────────────────────────────────────────
    print("\nRunning final evaluation on training set …")
    mean_reward, std_reward = evaluate_policy(
        model, train_env, n_eval_episodes=50, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")

    train_env.close()


if __name__ == "__main__":
    main()
