from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from core.trainer import build_env
from core.utils import load_yaml


CONFIG_PATH = Path(__file__).with_name("config_debug_extractor.yaml")
OUT_DIR = Path(__file__).with_name("debug_outputs")


def _assert_obs(env, obs, info) -> None:
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.int64
    assert obs.shape == (8,)
    assert env.observation_space.contains(obs)
    assert "symbolic_state" in info
    assert "extractor_debug" in info
    assert info["extractor_debug"].get("chicken_detected") is True


def _save_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, [np.asarray(frame, dtype=np.uint8) for frame in frames], duration=1.0 / fps, loop=0)


def main() -> None:
    try:
        config = load_yaml(CONFIG_PATH)
        env = build_env(config)
    except ImportError as exc:
        print(f"SKIP: Atari/ALE dependencies unavailable: {exc}")
        return

    seeds = [0, 1, 2, 42, 10000]
    state_mins = np.full(8, 10**9, dtype=np.int64)
    state_maxs = np.full(8, -1, dtype=np.int64)
    frames_checked = 0
    positive_rewards = 0
    debug_paths: list[Path] = []
    total_fallback_count = 0

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            _assert_obs(env, obs, info)
            state_mins = np.minimum(state_mins, obs)
            state_maxs = np.maximum(state_maxs, obs)
            frames_checked += 1
            if seed == 0:
                first_png = OUT_DIR / "seed0_overlay_first.png"
                Image.fromarray(env.render(mode="rgb_array")).save(first_png)
                debug_paths.append(first_png)

            y_start = info["extractor_debug"]["chicken_center"][1]
            seed0_up_frames: list[np.ndarray] = []
            for action, count in [(0, 30), (1, 40), (2, 40)]:
                y_values: list[int] = []
                for _ in range(count):
                    obs, reward, terminated, truncated, info = env.step(action)
                    _assert_obs(env, obs, info)
                    assert isinstance(float(reward), float)
                    assert isinstance(terminated, bool)
                    assert isinstance(truncated, bool)
                    assert info["extractor_debug"]["failure_count"] == 0
                    total_fallback_count += int(bool(info["extractor_debug"].get("fallback_used", False)))
                    if float(reward) > 0.0:
                        positive_rewards += 1
                    y_values.append(int(info["extractor_debug"]["chicken_center"][1]))
                    state_mins = np.minimum(state_mins, obs)
                    state_maxs = np.maximum(state_maxs, obs)
                    frames_checked += 1
                    if seed == 0 and action == 1:
                        seed0_up_frames.append(env.render(mode="rgb_array"))
                    if terminated or truncated:
                        obs, info = env.reset(seed=seed)
                        _assert_obs(env, obs, info)
                if action == 1 and y_values and min(y_values) > y_start:
                    print(f"WARN: seed {seed} UP script did not move chicken upward; possible collision/no-op")
                if action == 2 and y_values and max(y_values) < min(y_start + 2, 209):
                    print(f"WARN: seed {seed} DOWN script did not move chicken downward; possible collision/no-op")
            if seed == 0 and seed0_up_frames:
                up_gif = OUT_DIR / "seed0_scripted_up.gif"
                _save_gif(seed0_up_frames, up_gif)
                debug_paths.append(up_gif)

        always_up_csv = OUT_DIR / "seed0_always_up_trace.csv"
        always_up_frames: list[np.ndarray] = []
        always_up_y: list[int] = []
        always_up_lanes: list[int] = []
        always_up_y_bins: list[int] = []
        always_up_positive_rewards = 0
        obs, info = env.reset(seed=0)
        _assert_obs(env, obs, info)
        always_up_start_y = int(info["extractor_debug"]["chicken_center"][1])
        with always_up_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "action", "reward", "chicken_x", "chicken_y", "state"])
            for step in range(1, 1001):
                obs, reward, terminated, truncated, info = env.step(1)
                _assert_obs(env, obs, info)
                assert info["extractor_debug"]["failure_count"] == 0
                total_fallback_count += int(bool(info["extractor_debug"].get("fallback_used", False)))
                center = info["extractor_debug"]["chicken_center"]
                always_up_y.append(int(center[1]))
                always_up_lanes.append(int(obs[0]))
                always_up_y_bins.append(int(obs[1]))
                if float(reward) > 0.0:
                    always_up_positive_rewards += 1
                    positive_rewards += 1
                state_mins = np.minimum(state_mins, obs)
                state_maxs = np.maximum(state_maxs, obs)
                frames_checked += 1
                if step <= 240:
                    always_up_frames.append(env.render(mode="rgb_array"))
                writer.writerow([step, 1, float(reward), int(center[0]), int(center[1]), " ".join(map(str, obs.tolist()))])
                if terminated or truncated:
                    break

        assert always_up_positive_rewards > 0, "Always-UP policy did not produce a positive Freeway reward"
        assert min(always_up_y) <= always_up_start_y - 80, (
            f"Detected chicken did not move upward enough: start={always_up_start_y}, min={min(always_up_y)}"
        )
        assert len(set(always_up_y)) > 20, "Detected chicken y is stuck"
        assert not (set(always_up_lanes) == {0} and set(always_up_y_bins) == {0}), (
            "Detected chicken is stuck at lane 0/y-bin 0"
        )
        always_up_gif = OUT_DIR / "seed0_always_up.gif"
        _save_gif(always_up_frames, always_up_gif)
        debug_paths.extend([always_up_csv, always_up_gif])

        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=123)
        random_frames: list[np.ndarray] = []
        for step in range(300):
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            _assert_obs(env, obs, info)
            assert info["extractor_debug"]["failure_count"] == 0
            total_fallback_count += int(bool(info["extractor_debug"].get("fallback_used", False)))
            if float(reward) > 0.0:
                positive_rewards += 1
            state_mins = np.minimum(state_mins, obs)
            state_maxs = np.maximum(state_maxs, obs)
            frames_checked += 1
            if step < 120:
                random_frames.append(env.render(mode="rgb_array"))
            if terminated or truncated:
                obs, info = env.reset(seed=123 + step)
                _assert_obs(env, obs, info)
        random_gif = OUT_DIR / "seed0_random_rollout.gif"
        _save_gif(random_frames, random_gif)
        debug_paths.append(random_gif)

        print("Freeway extractor validation summary")
        print(f"frames_checked={frames_checked}")
        print("extractor_failures=0")
        print(f"fallback_count={total_fallback_count}")
        print(f"state_min={state_mins.tolist()}")
        print(f"state_max={state_maxs.tolist()}")
        print(f"positive_reward_events={positive_rewards}")
        print(f"always_up_positive_reward_events={always_up_positive_rewards}")
        print("debug_outputs:")
        for path in debug_paths:
            print(f"  {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
