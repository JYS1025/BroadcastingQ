"""Action-direction probe for the SO101 IK reach env.

Sanity check that each discrete Cartesian action actually moves the end-effector
in the expected direction of the observed base-frame ``target_error_xyz``.

For action ``+x`` we EXPECT ``target_error_x`` to decrease (EE moves +x toward a
fixed target, so target - ee shrinks on x) while y/z stay roughly unchanged. If
instead +x mostly changes a different axis, or moves error the wrong way, the
action frame is misaligned with the observation frame -- which makes the task
unlearnable for any tabular method, regardless of RL hyperparameters.

Run (headless):
    /workspace/isaaclab/isaaclab.sh -p -m applications.so101_reach.probe_actions \
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
    parser = argparse.ArgumentParser(description="Probe action->error directions for SO101 IK reach.")
    parser.add_argument("--config", required=True, help="Path to a SO101 reach IK config.")
    parser.add_argument("--resets", type=int, default=8, help="Number of random resets to average over.")
    parser.add_argument("--steps", type=int, default=5, help="Consecutive steps to hold each action.")
    parser.add_argument("--settle", type=int, default=15, help="Noop steps after reset before probing (lets the arm reach gravity equilibrium).")
    parser.add_argument("--reach_offset", type=float, default=0.06, help="Fixed target offset (m) for the per-axis reachability stress test.")
    parser.add_argument("--reach_hold", type=int, default=60, help="Sim steps to hold the fixed reach-test target (tests convergence vs. kinematic limit).")
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
    # Hold a fixed target while probing so error change reflects EE motion only.
    config["env"]["terminate_on_success"] = False

    set_global_seed(int(config.get("seed", 0)))
    env = build_env(config)
    action_names = env.action_names
    n_actions = env.action_space.n

    noop_action = action_names.index("noop") if "noop" in action_names else None

    def settle(env, n):
        """Hold the current pose for n steps so the arm reaches gravity equilibrium."""
        for _ in range(int(n)):
            if noop_action is not None:
                env.step(int(noop_action))

    try:
        # accumulate mean error-delta vector per action across resets
        sums = {a: np.zeros(3, dtype=np.float64) for a in range(n_actions)}
        counts = {a: 0 for a in range(n_actions)}
        # characterize the post-settle drift over the probe window (should be ~0 if the
        # arm holds): measured from the noop action.
        for reset_idx in range(int(args.resets)):
            for action in range(n_actions):
                env.reset(seed=1000 + reset_idx)
                settle(env, args.settle)
                err0 = np.asarray(env._get_info()["target_error_xyz"], dtype=np.float64)
                for _ in range(int(args.steps)):
                    _, _, terminated, truncated, info = env.step(int(action))
                    if terminated or truncated:
                        break
                err1 = np.asarray(info["target_error_xyz"], dtype=np.float64)
                sums[action] += (err1 - err0)
                counts[action] += 1

        means = {a: sums[a] / max(counts[a], 1) for a in range(n_actions)}
        baseline = means[noop_action] if noop_action is not None else np.zeros(3)

        print("\n" + "=" * 88)
        print("ACTION -> mean change in target_error_xyz  (settle=%d, then %d steps, over %d resets)"
              % (args.settle, args.steps, args.resets))
        print("Raw = absolute change; Net = raw minus noop drift (the action's true effect).")
        print("Expectation: '+x' => Net d(err_x) < 0 and dominates; same for y,z.")
        print("=" * 88)
        print("noop drift over window (should be ~0 if the arm holds): "
              f"[{baseline[0]:+.5f} {baseline[1]:+.5f} {baseline[2]:+.5f}]")
        print("-" * 88)
        print(f"{'action':>8} | {'rawX':>9} {'rawY':>9} {'rawZ':>9} | {'netX':>9} {'netY':>9} {'netZ':>9} | verdict")
        print("-" * 88)
        axis_of = {"x": 0, "y": 1, "z": 2}
        for action in range(n_actions):
            name = action_names[action]
            raw = means[action]
            net = raw - baseline
            verdict = ""
            if name != "noop" and len(name) == 2:
                sign, axis = name[0], name[1]
                ai = axis_of.get(axis)
                if ai is not None:
                    expected_sign = -1.0 if sign == "+" else 1.0
                    dominant = int(np.argmax(np.abs(net)))
                    ok_axis = dominant == ai
                    ok_sign = np.sign(net[ai]) == expected_sign and abs(net[ai]) > 1e-4
                    verdict = "OK" if (ok_axis and ok_sign) else "*** MISMATCH ***"
            print(f"{name:>8} | {raw[0]:>9.5f} {raw[1]:>9.5f} {raw[2]:>9.5f} | "
                  f"{net[0]:>9.5f} {net[1]:>9.5f} {net[2]:>9.5f} | {verdict}")
        print("=" * 88)
        print("If Net columns are correct but noop drift is large, the arm is sagging under")
        print("gravity -> increase env.reset_settle_steps (transient) or revisit control authority.")

        # --- Reachability stress test -------------------------------------------------
        # Hold a LARGE fixed absolute target per axis for many sim steps. If the EE
        # converges close to the offset, the weak axes just needed more substeps. If it
        # asymptotes far short -- especially with an IK joint pinned at a limit -- the arm
        # is kinematically stuck (workspace/limit), which is a posture problem, not RL.
        def joint_report():
            try:
                robot = env._robot
                names = list(robot.data.joint_names)
                q = robot.data.joint_pos[0].detach().cpu().numpy()
                lim = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
                flags = []
                for nm, qi, (lo, hi) in zip(names, q, lim):
                    frac = (qi - lo) / (hi - lo + 1e-9)
                    tag = "LIMIT_LO" if frac < 0.03 else ("LIMIT_HI" if frac > 0.97 else "")
                    if tag:
                        flags.append(f"{nm}={qi:+.2f}[{tag}]")
                return ", ".join(flags) if flags else "(no joint near limit)"
            except Exception as exc:  # defensive: API differences shouldn't crash the probe
                return f"(joint introspection unavailable: {exc})"

        torch = env._torch
        off = float(args.reach_offset)
        print("\n" + "=" * 88)
        print("REACHABILITY STRESS TEST: hold fixed target = settled_ee +/- %.3f m, %d sim steps"
              % (off, args.reach_hold))
        print("=" * 88)
        print(f"{'target':>8} | {'achieved (m)':>14} | {'frac':>6} | joints at limit")
        print("-" * 88)
        for axis_name, ai in (("x", 0), ("y", 1), ("z", 2)):
            for sign in (+1.0, -1.0):
                env.reset(seed=2000)
                settle(env, args.settle)
                ee0 = env._ee_pos_b().clone()
                target = ee0.clone()
                target[0, ai] += sign * off
                for _ in range(int(args.reach_hold)):
                    env.env.step(target)
                ee1 = env._ee_pos_b()
                moved = float((ee1 - ee0)[0, ai].detach().cpu().item())
                frac = moved / (sign * off)
                label = f"{'+' if sign > 0 else '-'}{axis_name}"
                print(f"{label:>8} | {moved:>14.5f} | {frac:>6.2f} | {joint_report()}")
        print("=" * 88)
        print("frac ~1.0 => reachable (just needed substeps).  frac << 1 with a joint at a")
        print("limit => kinematically stuck: fix the robot's operating posture, not the agent.")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
