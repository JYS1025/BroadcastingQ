from __future__ import annotations

from pathlib import Path

import numpy as np

from core.seeding import set_global_seed
from core.trainer import build_agent, build_env, build_visualizer
from core.utils import load_yaml


APP_DIR = Path(__file__).parent
CONFIG_PATHS = sorted(APP_DIR.glob("config_*.yaml"))


def main() -> None:
    for path in CONFIG_PATHS:
        _validate_config(path)
    for path in sorted(APP_DIR.glob("config_sbq*.yaml")):
        _validate_structural(path)
    print("minigrid_lavagap validation passed")


def _validate_config(path: Path) -> None:
    config = load_yaml(path)
    env = build_env(config)
    visualizer = build_visualizer(config)
    rng = np.random.default_rng(0)
    frames = []
    try:
        for seed in [0, 1, 42]:
            obs, info = env.reset(seed=seed)
            _assert_obs_info(env, obs, info)
            frames.append(env.render(mode="rgb_array"))
            for _ in range(20):
                action = env.action_space.sample(rng)
                assert env.action_space.contains(action)
                obs, reward, terminated, truncated, info = env.step(action)
                _assert_step(env, obs, reward, terminated, truncated, info)
                frames.append(env.render(mode="rgb_array"))
                if terminated or truncated:
                    break
        _save_debug(visualizer, frames, APP_DIR / "debug_outputs")
    finally:
        env.close()


def _validate_structural(path: Path) -> None:
    config = load_yaml(path)
    rng = set_global_seed(int(config.get("seed", 0)))
    env = build_env(config)
    try:
        obs, _ = env.reset(seed=0)
        agent = build_agent(config, env, rng)
        state = env.observation_space.to_index(obs)
        assert agent.get_distance(state, state) == 0.0
        neighbors = agent.get_neighborhood(state)
        weights = agent.compute_kernel_weights(state, neighbors)
        assert len(neighbors) > 0
        assert np.all(np.isfinite(weights))
        assert np.isclose(float(np.sum(weights)), 1.0)
        center = env.observation_space.from_index(state)
        valid_positions = getattr(env.observation_space, "valid_agent_positions", None)
        for idx in neighbors:
            candidate = env.observation_space.from_index(int(idx))
            assert env.observation_space.contains(candidate)
            assert np.array_equal(candidate[3:], center[3:])
            pos = (int(candidate[0]), int(candidate[1]))
            if valid_positions is not None:
                assert pos in valid_positions
            assert agent.get_distance(state, int(idx)) <= float(agent.search_radius)
    finally:
        env.close()


def _assert_obs_info(env, obs, info: dict) -> None:
    assert obs.dtype == np.int64
    assert env.observation_space.contains(obs)
    assert "symbolic_state" in info
    assert "success" in info
    assert "action_name" in info


def _assert_step(env, obs, reward, terminated, truncated, info: dict) -> None:
    _assert_obs_info(env, obs, info)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def _save_debug(visualizer, frames, out_dir: Path) -> None:
    assert visualizer is not None
    assert frames
    frame = np.asarray(frames[-1])
    assert frame.ndim == 3 and frame.shape[-1] == 3
    visualizer.save_final_frame(frame, out_dir / "debug.png")
    visualizer.save_episode_gif(frames[:8], out_dir / "debug.gif", fps=4)


if __name__ == "__main__":
    main()
