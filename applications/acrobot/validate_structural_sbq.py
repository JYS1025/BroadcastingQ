from __future__ import annotations

from pathlib import Path

import numpy as np

from core.seeding import set_global_seed
from core.trainer import build_agent, build_env
from core.utils import load_yaml


CONFIG_PATH = Path(__file__).with_name("config_sbq_structural.yaml")


def main() -> None:
    config = load_yaml(CONFIG_PATH)
    rng = set_global_seed(int(config.get("seed", 0)))
    env = build_env(config)
    agent = build_agent(config, env, rng)
    try:
        obs, _ = env.reset(seed=int(config.get("seed", 0)))
        assert env.observation_space.contains(obs)
        state = env.observation_space.to_index(obs)
        assert agent.get_distance(state, state) == 0.0

        nvec = env.observation_space.nvec
        wrap_a = env.observation_space.to_index(np.array([0, 0, 0, 0], dtype=np.int64))
        wrap_b = env.observation_space.to_index(np.array([nvec[0] - 1, 0, 0, 0], dtype=np.int64))
        assert agent.get_distance(wrap_a, wrap_b) == 1.0
        vel_far = env.observation_space.to_index(np.array([0, 0, nvec[2] - 1, 0], dtype=np.int64))
        assert agent.get_distance(wrap_a, vel_far) == 0.5 * (nvec[2] - 1)

        neighbors = agent.get_neighborhood(state)
        weights = agent.compute_kernel_weights(state, neighbors)
        assert len(neighbors) > 0
        assert np.all(np.isfinite(weights))
        assert np.isclose(float(np.sum(weights)), 1.0)
        for idx in neighbors:
            candidate = env.observation_space.from_index(int(idx))
            assert env.observation_space.contains(candidate)
            assert agent.get_distance(state, int(idx)) <= float(agent.search_radius) + 1e-9

        for _ in range(5):
            action = env.action_space.sample(rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(float(reward), float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(100000)))

        for _ in range(200):
            action = env.action_space.sample(rng)
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        print("Acrobot structural SBQ validation passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
