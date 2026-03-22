#!/usr/bin/env python3
"""
Play script for a trained Stable-Baselines3 DQN Atari agent.

Usage examples:
    python play.py
    python play.py --model-path "Models/Carine/10_Optimized.zip" --env-id ALE/Breakout-v5 --episodes 3
    python play.py --model-path auto --env-id ALE/Breakout-v5 --episodes 3

Notes:
- In SB3, greedy action selection is achieved with `deterministic=True` in `model.predict(...)`.
- Keep `--env-id` identical to the one used during training.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple


def _import_dependencies() -> tuple[Any, Any, Any, Any, Any]:
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
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecEnv, VecFrameStack
        return np, gym, DQN, make_atari_env, VecEnv, VecFrameStack
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install them first, for example:\n"
            "pip install \"gymnasium[atari,accept-rom-license]\" stable-baselines3[extra] ale-py numpy imageio\n\n"
            "If using Colab, run this install command in a notebook cell before executing play.py."
        ) from exc


def _build_train_like_vec_atari_env(
    env_id: str,
    seed: Optional[int],
    n_stack: int,
    render_mode: str,
    make_atari_env: Any,
    VecFrameStack: Any,
) -> Any:
    """Build environment using the same pipeline as train.py."""
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": render_mode},
    )
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def _select_compatible_env(
    model: Any,
    env_id: str,
    seed: Optional[int],
    n_stack: int,
    render_mode: str,
    make_atari_env: Any,
    VecFrameStack: Any,
) -> Tuple[object, str]:
    builders: list[Tuple[str, Callable[[], object]]] = [
        (
            "Vec Atari + FrameStack (train.py-compatible)",
            lambda: _build_train_like_vec_atari_env(
                env_id=env_id,
                seed=seed,
                n_stack=n_stack,
                render_mode=render_mode,
                make_atari_env=make_atari_env,
                VecFrameStack=VecFrameStack,
            ),
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


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _find_best_from_csv(csv_path: Path, model_col: str, reward_col: str) -> Optional[tuple[str, float]]:
    if not csv_path.exists():
        return None

    best_name: Optional[str] = None
    best_reward = float("-inf")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items() if k is not None}
            model_name = cleaned.get(model_col, "")
            reward = _safe_float(cleaned.get(reward_col))
            if not model_name or reward is None:
                continue
            if reward > best_reward:
                best_name = model_name
                best_reward = reward

    if best_name is None:
        return None
    return best_name, best_reward


def _find_zip_for_model(member_dir: Path, model_name: str) -> Optional[Path]:
    exact = member_dir / f"{model_name}.zip"
    if exact.exists():
        return exact

    model_key = model_name.strip().lower()
    for p in member_dir.glob("*.zip"):
        stem = p.stem.strip().lower()
        if stem == model_key or stem.startswith(model_key):
            return p
    return None


def _resolve_best_model_from_models_dir(models_dir: Path) -> Optional[Path]:
    member_config = [
        {
            "folder": "Ayomide",
            "csv": "hyperparameter_log_results with observations.csv",
            "model_col": "Model Name",
            "reward_col": "Observed Mean Reward",
        },
        {
            "folder": "Armstrong",
            "csv": "hyperparameter_log_results with observations.csv",
            "model_col": "Model Name",
            "reward_col": "Mean Reward",
        },
        {
            "folder": "Carine",
            "csv": "hyperparameter_log_results with observations.csv",
            "model_col": "Model Name",
            "reward_col": "Observed Mean Reward",
        },
        {
            "folder": "Gustave",
            "csv": "hyperparameter_log_results.csv",
            "model_col": "Experiment",
            "reward_col": "Avg_Score",
        },
    ]

    best_path: Optional[Path] = None
    best_reward = float("-inf")

    for cfg in member_config:
        member_dir = models_dir / cfg["folder"]
        csv_path = member_dir / cfg["csv"]
        best = _find_best_from_csv(
            csv_path=csv_path,
            model_col=cfg["model_col"],
            reward_col=cfg["reward_col"],
        )
        if best is None:
            continue

        model_name, reward = best
        model_zip = _find_zip_for_model(member_dir, model_name)
        if model_zip is None:
            continue

        if reward > best_reward:
            best_reward = reward
            best_path = model_zip

    return best_path


def _resolve_model_path(model_path_arg: str, models_dir_arg: str) -> Path:
    candidate = Path(model_path_arg)
    if model_path_arg.lower() != "auto" and candidate.exists():
        return candidate

    models_dir = Path(models_dir_arg)
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}. "
            "Pass --model-path explicitly or provide a valid --models-dir."
        )

    best = _resolve_best_model_from_models_dir(models_dir)
    if best is None:
        if model_path_arg.lower() != "auto":
            raise FileNotFoundError(
                f"Model file not found: {model_path_arg}, and auto-detection could not find a valid best model."
            )
        raise FileNotFoundError(
            "Could not auto-detect a best model zip from Models/ logs. Pass --model-path explicitly."
        )
    return best


def _prepare_numpy_pickle_compat(np: Any) -> None:
    """
    Compatibility shim:
    models saved with NumPy 2 may reference `numpy._core.*` during unpickling.
    On environments pinned to NumPy 1.x (for older Torch builds), we alias
    these module paths so model loading can proceed.
    """
    try:
        core_mod = np.core
        numeric_mod = np.core.numeric
        if "numpy._core" not in sys.modules:
            sys.modules["numpy._core"] = core_mod
        if "numpy._core.numeric" not in sys.modules:
            sys.modules["numpy._core.numeric"] = numeric_mod
    except Exception:
        # Non-fatal: if this fails, normal loading still applies.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and play a trained DQN Atari model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Models/Carine/10_Optimized.zip",
        help="Path to the trained model zip file (default: Models/Carine/10_Optimized.zip). Use 'auto' to pick from CSV logs.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="Models",
        help="Top-level Models directory used when --model-path auto-detection is enabled.",
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

    np, gym, DQN, make_atari_env, VecEnv, VecFrameStack = _import_dependencies()
    _prepare_numpy_pickle_compat(np)

    resolved_model_path = _resolve_model_path(args.model_path, args.models_dir)

    seed = None if args.seed < 0 else args.seed

    env = None
    try:
        env = _build_train_like_vec_atari_env(
            env_id=args.env_id,
            seed=seed,
            n_stack=args.n_stack,
            render_mode=args.render_mode,
            make_atari_env=make_atari_env,
            VecFrameStack=VecFrameStack,
        )
        env_mode = "Vec Atari + FrameStack (train.py-compatible)"

        obs_space_for_load = env.observation_space
        obs_shape = getattr(obs_space_for_load, "shape", None)
        if isinstance(obs_shape, tuple) and len(obs_shape) == 3 and obs_shape[-1] == args.n_stack:
            # Saved CNN models are often channels-first internally.
            obs_space_for_load = gym.spaces.Box(
                low=0,
                high=255,
                shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
                dtype=env.observation_space.dtype,
            )

        print(f"Loading model from: {resolved_model_path}")
        model = DQN.load(
            str(resolved_model_path),
            custom_objects={
                "observation_space": obs_space_for_load,
                "action_space": env.action_space,
                "_last_obs": None,
                "_last_episode_starts": None,
                "_last_original_obs": None,
                "ep_info_buffer": [],
                "ep_success_buffer": [],
                "lr_schedule": lambda _: 1e-4,
                "exploration_schedule": lambda _: 0.02,
                "train_freq": (4, "step"),
            },
        )
        model.set_env(env)
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
