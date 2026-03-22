#!/usr/bin/env python3
import os
import json
import argparse
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def load_config(config_path, exp_id):
    with open(config_path, 'r') as f:
        configs = json.load(f)
    # Find the specific experiment by ID
    for cfg in configs:
        if cfg['id'] == exp_id:
            return cfg
    raise ValueError(f"Experiment ID {exp_id} not found in {config_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiments.json")
    parser.add_argument("--exp-id", type=int, required=True, help="Index of the experiment to run")
    parser.add_argument("--env-id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--steps", type=int, default=200000)
    args = parser.parse_args()

    # Load parameters from external JSON
    cfg = load_config(args.config, args.exp_id)

    print(f"Running Experiment: {cfg['name']}")

    # Setup Environment
    env = make_atari_env(args.env_id, n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # Initialize Model with JSON parameters
    model = DQN(
        cfg['policy'],
        env,
        learning_rate=cfg['lr'],
        gamma=cfg['gamma'],
        batch_size=cfg['batch_size'],
        buffer_size=10000, # Kept low for memory safety
        exploration_initial_eps=cfg['epsilon_start'],
        exploration_final_eps=cfg['epsilon_end'],
        exploration_fraction=cfg['epsilon_decay'],
        verbose=1,
        tensorboard_log="./logs/"
    )

    # Train and Save
    model.learn(total_timesteps=args.steps, log_interval=100)
    
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{cfg['name']}")
    print(f"Success: {cfg['name']} saved.")

if __name__ == "__main__":
    main()