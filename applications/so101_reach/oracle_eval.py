"""Oracle controller for the SO101 IK reach env.

Runs a hand-coded greedy controller (no learning): each step, move along the axis
with the largest target-error, in the direction that reduces it. This measures the
TASK's solvability ceiling, independent of any RL agent:

- If the oracle reaches the goal reliably, the env is solvable and the learning
  failure is an RL/state/reward problem (e.g. the binned target_error state is not
  Markov because it omits arm posture).
- If the oracle CANNOT solve it, the task is too hard with these discrete actions
  (z under-actuated / action coupling) and must be made easier before RL can work.

Run (headless):
    /workspace/isaaclab/isaaclab.sh -p -m applications.so101_reach.oracle_eval \
        --config applications/so101_reach/config_ik_sbq.yaml --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand-coded oracle reach controller for SO101 IK env.")
    parser.add_argument("--config", required=True, help="Path to a SO101 reach IK config.")
    parser.add_argument("--episodes", type=int, default=40, help="Number of oracle rollouts.")
    parser.add_argument("--max_steps", type=int, default=80, help="Max steps per episode.")
    parser.add_argument("--deadband", type=float, default=0.005, help="Axis error below this is treated as solved (pick next-largest, or noop).")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from applications.so101_reach.runner import build_env
    from core.seeding import set_global_seed
    from core.utils import load_yaml

    config = load_yaml(args.config)
    config.setdefault("env", {})
    config["env"]["device"] = args.device
    config["env"]["use_fabric"] = True

    set_global_seed(int(config.get("seed", 0)))
    env = build_env(config)
    action_names = env.action_names

    def oracle_action(target_error):
        # err = target - ee. To shrink err on an axis, move EE toward the target:
        # +axis if err>0, -axis if err<0. Act on the largest-magnitude axis.
        err = np.asarray(target_error, dtype=np.float64)
        mag = np.abs(err)
        ai = int(np.argmax(mag))
        if mag[ai] < args.deadband:
            return action_names.index("noop") if "noop" in action_names else 0
        axis = "xyz"[ai]
        sign = "+" if err[ai] > 0 else "-"
        name = f"{sign}{axis}"
        return action_names.index(name)

    try:
        successes = 0
        final_dists = []
        lengths = []
        start_dists = []
        for ep in range(int(args.episodes)):
            _, info = env.reset(seed=5000 + ep)
            start_dists.append(float(info["distance_to_target"]))
            done = False
            length = 0
            success = False
            while not done and length < int(args.max_steps):
                a = oracle_action(info["target_error_xyz"])
                _, _, terminated, truncated, info = env.step(int(a))
                length += 1
                success = bool(info.get("success", False))
                done = bool(terminated or truncated)
            successes += int(success)
            final_dists.append(float(info["distance_to_target"]))
            lengths.append(length)

        sr = successes / max(int(args.episodes), 1)
        print("\n" + "=" * 72)
        print("ORACLE (largest-error-axis greedy) over %d episodes" % args.episodes)
        print("=" * 72)
        print(f"success_rate     : {sr:.2%}  ({successes}/{args.episodes})")
        print(f"start distance   : mean {np.mean(start_dists):.4f}  (shell sampling range)")
        print(f"final distance   : mean {np.mean(final_dists):.4f}  median {np.median(final_dists):.4f}")
        print(f"episode length   : mean {np.mean(lengths):.1f}")
        print("=" * 72)
        print("High success => env is solvable; RL failure is a state/reward problem")
        print("(binned target_error likely non-Markov: add arm posture to the state).")
        print("Low success  => task too hard with these actions; make it easier first")
        print("(smaller ee_relative radius, looser success_threshold, or fix z authority).")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
