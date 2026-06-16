from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SO101 reach checkpoint in Isaac Lab.")
    parser.add_argument("--config", required=True, help="Path to a SO101 reach experiment config.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path, or 'latest' under the config run dir.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of greedy evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per episode; defaults to training.eval_max_steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment/eval seed.")
    parser.add_argument("--task", type=str, default=None, help="Override the Isaac task id.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of envs; only 1 is supported.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Isaac Fabric.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from applications.so101_reach.runner import build_agent, build_env
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

    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    env = None
    try:
        env = build_env(config)
        agent = build_agent(config, env, rng)
        checkpoint = _resolve_checkpoint(config, args.checkpoint)
        agent.load(str(checkpoint))
        if hasattr(agent, "epsilon"):
            agent.epsilon = 0.0

        episodes = int(args.episodes)
        max_steps = int(config.get("training", {}).get("eval_max_steps", 360) if args.max_steps is None else args.max_steps)
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        distances: list[float] = []

        print(f"checkpoint: {checkpoint}", flush=True)
        print(f"episodes: {episodes}", flush=True)
        print(f"max_steps: {max_steps}", flush=True)
        for episode_idx in range(episodes):
            obs, info = env.reset(seed=seed + episode_idx)
            episode_return = 0.0
            length = 0
            done = False
            while not done and length < max_steps:
                if not args.headless and not simulation_app.is_running():
                    done = True
                    break
                action = agent.act(obs, explore=False)
                obs, reward, terminated, truncated, info = env.step(int(action))
                episode_return += float(reward)
                length += 1
                done = bool(terminated or truncated)
            success = float(bool(info.get("success", False)))
            distance = float(info.get("distance_to_target", np.nan))
            returns.append(episode_return)
            lengths.append(length)
            successes.append(success)
            distances.append(distance)
            print(
                f"episode={episode_idx} return={episode_return:.4f} length={length} "
                f"success={bool(success)} distance={distance:.5f}",
                flush=True,
            )

        print("summary", flush=True)
        print(f"return_mean={float(np.mean(returns)):.4f}", flush=True)
        print(f"return_std={float(np.std(returns)):.4f}", flush=True)
        print(f"length_mean={float(np.mean(lengths)):.2f}", flush=True)
        print(f"success_rate={float(np.mean(successes)):.4f}", flush=True)
        print(f"distance_mean={float(np.nanmean(distances)):.5f}", flush=True)
        print(f"distance_min={float(np.nanmin(distances)):.5f}", flush=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _resolve_checkpoint(config: dict, checkpoint: str) -> Path:
    if checkpoint != "latest":
        return Path(checkpoint)
    logging_cfg = config.get("logging", {})
    output_root = Path(logging_cfg.get("output_root", "outputs"))
    run_name = str(logging_cfg.get("run_name", f"so101_reach_{config['agent']['name']}_seed{config.get('seed', 0)}"))
    checkpoint_dir = output_root / run_name / "checkpoints"
    agent_name = str(config.get("agent", {}).get("name", "")).lower()
    suffixes = [".pt"] if agent_name == "dqn" else [".pkl", ".pkl.npy"]
    candidates = []
    for suffix in suffixes:
        candidates.extend(checkpoint_dir.glob(f"agent_step_*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return max(candidates, key=_checkpoint_step)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"agent_step_(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


if __name__ == "__main__":
    main()
