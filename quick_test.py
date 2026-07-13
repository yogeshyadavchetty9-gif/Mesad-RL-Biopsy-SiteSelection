"""
quick_test.py — Sanity-check MESADEnv without training.

Run this first to verify your dataset paths are correct and the
environment works end-to-end before launching a full training run.

Usage
-----
    python quick_test.py \
        --train_images mesad-real/train/images \
        --train_annots mesad-real/train/annotations
"""

import argparse
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mesad_env import MESADEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_images", required=True)
    p.add_argument("--train_annots", required=True)
    p.add_argument("--grid_cols", type=int, default=4)
    p.add_argument("--grid_rows", type=int, default=4)
    p.add_argument("--obs_size",  type=int, default=128)
    p.add_argument("--n_steps",   type=int, default=50,
                   help="Number of random steps to run")
    return p.parse_args()


def main():
    args = parse_args()

    print("Building MESADEnv …")
    env = MESADEnv(
        image_dir      = args.train_images,
        annotation_dir = args.train_annots,
        grid_cols      = args.grid_cols,
        grid_rows      = args.grid_rows,
        obs_w          = args.obs_size,
        obs_h          = args.obs_size,
    )

    print(f"  Dataset size      : {len(env.index)} images")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Action space      : {env.action_space}")
    print(f"  Grid              : {env.grid_cols}×{env.grid_rows}"
          f" = {env.grid_cols * env.grid_rows} cells\n")

    # SB3 environment compliance check
    print("Running SB3 env_checker …")
    check_env(env, warn=True)
    print("  PASSED\n")

    # Random rollout
    print(f"Running {args.n_steps} random steps …")
    obs, info = env.reset()
    total_reward = 0.0
    episodes     = 0
    ep_rewards   = []
    ep_r         = 0.0

    for i in range(args.n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_r        += reward
        total_reward += reward

        done = terminated or truncated
        if done:
            ep_rewards.append(ep_r)
            ep_r = 0.0
            episodes += 1
            obs, info = env.reset()

    if episodes == 0:
        episodes = 1
        ep_rewards.append(ep_r)

    print(f"  Steps run         : {args.n_steps}")
    print(f"  Episodes          : {episodes}")
    print(f"  Mean episode reward : {np.mean(ep_rewards):.3f}")
    print(f"  Last info         : {info}")
    print("\nQuick test PASSED — environment is ready for training.")
    env.close()


if __name__ == "__main__":
    main()
