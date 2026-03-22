#!/usr/bin/env python3
"""
select_best_model.py
--------------------
Reads each group member's hyperparameter_log_results CSV file, identifies
the best model per member (highest Observed Mean Reward / Avg_Score), then
identifies the single best model across the whole group.

Contributed by: Ayomide (AgbajeCity)

Usage:
    python select_best_model.py
        python select_best_model.py --models-dir Models --reward-col "Observed Mean Reward"
"""

from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration: member folders and their CSV filenames
# ---------------------------------------------------------------------------
MEMBER_CONFIG = [
      {
                "member": "Ayomide",
                "folder": "Ayomide",
                "csv": "hyperparameter_log_results with observations.csv",
                "model_col": "Model Name",
                "reward_col": "Observed Mean Reward",
      },
      {
                "member": "Armstrong",
                "folder": "Armstrong",
                "csv": "hyperparameter_log_results with observations.csv",
                "model_col": "Model Name",
                "reward_col": "Mean Reward",
      },
      {
                "member": "Carine",
                "folder": "Carine",
                "csv": "hyperparameter_log_results with observations.csv",
                "model_col": "Model Name",
                "reward_col": "Observed Mean Reward",
      },
      {
                "member": "Gustave",
                "folder": "Gustave",
                "csv": "hyperparameter_log_results.csv",
                "model_col": "Experiment",
                "reward_col": "Avg_Score",
      },
]


def _safe_float(val: str) -> Optional[float]:
      """Convert a string to float, returning None on failure."""
      try:
                return float(val.strip())
except (ValueError, AttributeError):
        return None


def find_best_in_csv(csv_path: Path, model_col: str, reward_col: str) -> Optional[dict]:
      """
          Parse a CSV file and return the row with the highest reward value.
              Returns None if the file cannot be parsed or no valid rows exist.
                  """
      if not csv_path.exists():
                print(f"  [WARNING] CSV not found: {csv_path}")
                return None

      best: Optional[dict] = None
      best_reward: float = float("-inf")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
              reader = csv.DictReader(f)
              for row in reader:
                            # Strip BOM / whitespace from keys
                            row = {k.strip(): v.strip() for k, v in row.items() if k}

                  name = row.get(model_col, "").strip()
            reward_str = row.get(reward_col, "").strip()

            if not name or not reward_str:
                              continue

            reward = _safe_float(reward_str)
            if reward is None:
                              continue

            if reward > best_reward:
                              best_reward = reward
                              best = {"model": name, "reward": reward, "row": row}

    return best


def find_best_model_zip(models_dir: Path, member_folder: str, model_name: str) -> Optional[Path]:
      """
          Locate the .zip file for a given model name inside the member's folder.
              Tries exact match first, then a case-insensitive / prefix search.
                  """
    folder = models_dir / member_folder
    if not folder.exists():
              return None

    model_name_clean = model_name.strip().lstrip()

    # Exact match
    candidate = folder / f"{model_name_clean}.zip"
    if candidate.exists():
              return candidate

    # Case-insensitive prefix scan
    for p in folder.iterdir():
              if p.suffix.lower() == ".zip":
                            stem = p.stem.lower().strip()
                            if stem == model_name_clean.lower() or stem.startswith(model_name_clean.lower()):
                return p

    return None


def main() -> None:
      parser = argparse.ArgumentParser(
                description="Identify the best DQN model across all group members."
      )
    parser.add_argument(
              "--models-dir",
              type=str,
              default="Models",
              help="Path to the top-level Models directory (default: Models)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
              raise FileNotFoundError(
                  f"Models directory not found: {models_dir.resolve()}\n"
                  "Run this script from the repo root, or pass --models-dir <path>."
    )

    print("=" * 60)
    print("  Deep Q-Learning Formative 3 — Best Model Selector")
    print("  Contributed by: Ayomide (AgbajeCity)")
    print("=" * 60)
    print()

    group_best: Optional[dict] = None
    group_best_reward: float = float("-inf")

    for cfg in MEMBER_CONFIG:
              member = cfg["member"]
        csv_path = models_dir / cfg["folder"] / cfg["csv"]
        print(f"── {member}")
        print(f"   CSV: {csv_path}")

        result = find_best_in_csv(
                      csv_path=csv_path,
                      model_col=cfg["model_col"],
                      reward_col=cfg["reward_col"],
        )

        if result is None:
                      print(f"   [SKIP] Could not determine best model for {member}.")
                      print()
                      continue

        model_name = result["model"]
        reward = result["reward"]
        zip_path = find_best_model_zip(models_dir, cfg["folder"], model_name)

        print(f"   Best model : {model_name}")
        print(f"   Reward     : {reward}")
        print(f"   Zip path   : {zip_path if zip_path else '(not found locally)'}")
        print()

        if reward > group_best_reward:
                      group_best_reward = reward
                      group_best = {
                          "member": member,
                          "model": model_name,
                          "reward": reward,
                          "zip_path": zip_path,
                          "folder": cfg["folder"],
                      }

    # ------------------------------------------------------------------
    # Report group winner
    # ------------------------------------------------------------------
    print("=" * 60)
    if group_best:
              print("  GROUP BEST MODEL")
        print(f"  Member : {group_best['member']}")
        print(f"  Model  : {group_best['model']}")
        print(f"  Reward : {group_best['reward']}")
        zip_path = group_best["zip_path"]
        if zip_path:
                      rel = os.path.relpath(zip_path)
                      print(f"  Zip    : {rel}")
                      print()
                      print("  To play this model, run:")
                      print(f"    python play.py --model-path \"{rel}\" --env-id ALE/Breakout-v5 --episodes 5")
else:
            print(f"  Zip    : (zip not found locally; download from GitHub)")
            member_folder = group_best["folder"]
            model_name = group_best["model"]
            print()
            print("  Once downloaded, run:")
            print(f"    python play.py --model-path \"Models/{member_folder}/{model_name}.zip\" --env-id ALE/Breakout-v5 --episodes 5")
else:
        print("  No valid models found. Check that CSV files exist and are correctly formatted.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("Individual Best Scores Summary")
    print("-" * 60)
    print(f"  {'Member':<15} {'Best Model':<30} {'Reward':>8}")
    print(f"  {'-'*15:<15} {'-'*30:<30} {'-'*8:>8}")
    for cfg in MEMBER_CONFIG:
              csv_path = models_dir / cfg["folder"] / cfg["csv"]
        result = find_best_in_csv(csv_path, cfg["model_col"], cfg["reward_col"])
        if result:
                      name = result["model"].strip().lstrip()[:29]
                      print(f"  {cfg['member']:<15} {name:<30} {result['reward']:>8.2f}")
else:
            print(f"  {cfg['member']:<15} {'N/A':<30} {'N/A':>8}")
    print("-" * 60)


if __name__ == "__main__":
    main()
