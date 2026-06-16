from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Q-learning agents on SO-ARM101 reach in Isaac Lab.")
    parser.add_argument("--config", required=True, help="Path to a SO101 reach experiment config.")
    parser.add_argument("--task", type=str, default=None, help="Override the Isaac task id.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs; only 1 is supported.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed.")
    parser.add_argument("--total_steps", type=int, default=None, help="Override training.total_steps.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Isaac Fabric.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from applications.so101_reach.runner import train
    from core.seeding import set_global_seed
    from core.utils import load_yaml

    config = load_yaml(args.config)
    config.setdefault("env", {})
    config["env"]["device"] = args.device
    config["env"]["use_fabric"] = not args.disable_fabric
    if args.task is not None:
        config["env"]["task"] = args.task
    if args.num_envs is not None:
        config["env"]["num_envs"] = args.num_envs
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_steps is not None:
        config.setdefault("training", {})["total_steps"] = args.total_steps

    set_global_seed(int(config.get("seed", 0)))
    try:
        output_dir = train(config, simulation_app=simulation_app)
        print(f"Run complete. Outputs saved to {output_dir}")
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"TRAINING FAILED: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

