#!/usr/bin/env python3
"""
Play script for a trained Stable-Baselines3 DQN Atari agent.

Usage examples:
    python play.py
    python play.py --model-path dqn_model.zip --env-id ALE/Breakout-v5 --episodes 3

Notes:
- In SB3, greedy action selection is achieved with `deterministic=True` in `model.predict(...)`.
- Keep `--env-id` identical to the one used during training.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable, Optional, Tuple


def _import_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import gymnasium as gym
        try:
            import ale_py
            gym.register_envs(ale_py)
        except Exception:
            # Older/newer Gymnasium setups may auto-register or use a different API.
            pass
        from stable_baselines3 import DQN
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage
        return np, gym, DQN, AtariWrapper, make_atari_env, VecEnv, VecFrameStack, VecTransposeImage
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install them first, for example:\n"
            "pip install \"gymnasium[atari,accept-rom-license]\" stable-baselines3[extra] ale-py numpy imageio\n\n"
            "If using Colab, run this install command in a notebook cell before executing play.py."
        ) from exc


def _build_vec_atari_env(
    env_id: str,
    seed: Optional[int],
    n_stack: int,
    render_mode: str,
    make_atari_env: Any,
    VecFrameStack: Any,
    VecTransposeImage: Any,
) -> Any:
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": render_mode},
        wrapper_kwargs={"terminal_on_life_loss": False},
    )
    env = VecFrameStack(env, n_stack=n_stack)
    env = VecTransposeImage(env)
    return env


def _build_atari_wrapper_env(env_id: str, seed: Optional[int], render_mode: str, gym: Any, AtariWrapper: Any) -> Any:
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env, terminal_on_life_loss=False)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _build_plain_env(env_id: str, seed: Optional[int], render_mode: str, gym: Any) -> Any:
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _select_compatible_env(
    model: Any,
    env_id: str,
    seed: Optional[int],
    n_stack: int,
    render_mode: str,
    gym: Any,
    AtariWrapper: Any,
    make_atari_env: Any,
    VecFrameStack: Any,
    VecTransposeImage: Any,
) -> Tuple[object, str]:
    builders: list[Tuple[str, Callable[[], object]]] = [
        (
            "Vec Atari + FrameStack + Transpose",
            lambda: _build_vec_atari_env(
                env_id=env_id,
                seed=seed,
                n_stack=n_stack,
                render_mode=render_mode,
                make_atari_env=make_atari_env,
                VecFrameStack=VecFrameStack,
                VecTransposeImage=VecTransposeImage,
            ),
        ),
        (
            "AtariWrapper (single env)",
            lambda: _build_atari_wrapper_env(
                env_id=env_id,
                seed=seed,
                render_mode=render_mode,
                gym=gym,
                AtariWrapper=AtariWrapper,
            ),
        ),
        (
            "Plain Gymnasium env",
            lambda: _build_plain_env(env_id=env_id, seed=seed, render_mode=render_mode, gym=gym),
        ),
    ]

    last_error: Optional[Exception] = None
    for label, build in builders:
        env = None
        try:
            env = build()
            model.set_env(env)
            return env, label
        except Exception as exc:
            last_error = exc
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    raise RuntimeError(
        "Could not create an environment compatible with this model. "
        "Make sure --env-id matches training and wrappers are consistent."
    ) from last_error


def _run_vec_eval(
    model: Any,
    env: Any,
    episodes: int,
    max_steps: int,
    sleep: float,
    capture_frames: bool,
) -> list[float]:
    rewards_per_episode: list[float] = []
    frames: list[Any] = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = [False]
        ep_reward = 0.0
        step_count = 0

        while not bool(done[0]) and step_count < max_steps:
            # GreedyQPolicy equivalent in SB3:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, _infos = env.step(action)
            frame = env.render()
            if capture_frames and frame is not None:
                frames.append(frame)
            ep_reward += float(rewards[0])
            done = dones
            step_count += 1
            if sleep > 0:
                time.sleep(sleep)

        rewards_per_episode.append(ep_reward)
        print(f"Episode {ep}/{episodes} | reward={ep_reward:.2f} | steps={step_count}")

    return rewards_per_episode, frames


def _run_single_env_eval(
    model: Any,
    env: Any,
    episodes: int,
    max_steps: int,
    sleep: float,
    seed: Optional[int],
    capture_frames: bool,
) -> list[float]:
    rewards_per_episode: list[float] = []
    frames: list[Any] = []

    for ep in range(1, episodes + 1):
        if seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)

        terminated, truncated = False, False
        ep_reward = 0.0
        step_count = 0

        while not (terminated or truncated) and step_count < max_steps:
            # GreedyQPolicy equivalent in SB3:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            frame = env.render()
            if capture_frames and frame is not None:
                frames.append(frame)
            ep_reward += float(reward)
            step_count += 1
            if sleep > 0:
                time.sleep(sleep)

        rewards_per_episode.append(ep_reward)
        print(f"Episode {ep}/{episodes} | reward={ep_reward:.2f} | steps={step_count}")

    return rewards_per_episode, frames


def _save_video(frames: list[Any], path: str, fps: int) -> None:
    if not frames:
        print("No frames captured. Skipping video export.")
        return
    try:
        import imageio
    except Exception as exc:
        raise RuntimeError(
            "imageio is required for --save-video. Install with: pip install imageio"
        ) from exc

    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved gameplay video: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and play a trained DQN Atari model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="dqn_model.zip",
        help="Path to the trained model zip file (default: dqn_model.zip)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="ALE/Breakout-v5",
        help="Gymnasium Atari environment ID used during training",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Max steps per episode (safety stop)",
    )
    parser.add_argument(
        "--n-stack",
        type=int,
        default=4,
        help="Frame stack for Vec Atari setup (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (set to -1 to disable)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep per step (seconds) to slow down playback",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Use 'human' for local GUI, 'rgb_array' for headless/Colab video capture",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Capture and save gameplay video (best with --render-mode rgb_array)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="gameplay.mp4",
        help="Output path for gameplay video",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="FPS for saved gameplay video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np, gym, DQN, AtariWrapper, make_atari_env, VecEnv, VecFrameStack, VecTransposeImage = _import_dependencies()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file not found: {args.model_path}. "
            "Place your dummy/trained model in this path or pass --model-path."
        )

    seed = None if args.seed < 0 else args.seed

    print(f"Loading model from: {args.model_path}")
    model = DQN.load(args.model_path)

    env = None
    try:
        env, env_mode = _select_compatible_env(
            model=model,
            env_id=args.env_id,
            seed=seed,
            n_stack=args.n_stack,
            render_mode=args.render_mode,
            gym=gym,
            AtariWrapper=AtariWrapper,
            make_atari_env=make_atari_env,
            VecFrameStack=VecFrameStack,
            VecTransposeImage=VecTransposeImage,
        )
        print(f"Using environment pipeline: {env_mode}")

        capture_frames = args.save_video

        if isinstance(env, VecEnv):
            rewards, frames = _run_vec_eval(
                model=model,
                env=env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                sleep=args.sleep,
                capture_frames=capture_frames,
            )
        else:
            rewards, frames = _run_single_env_eval(
                model=model,
                env=env,
                episodes=args.episodes,
                max_steps=args.max_steps,
                sleep=args.sleep,
                seed=seed,
                capture_frames=capture_frames,
            )

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        print("\nEvaluation complete")
        print(f"Episode rewards: {[round(r, 2) for r in rewards]}")
        print(f"Average reward: {avg_reward:.2f}")

        if args.save_video:
            _save_video(frames=frames, path=args.video_path, fps=args.video_fps)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
