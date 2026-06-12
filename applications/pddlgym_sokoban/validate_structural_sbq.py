from __future__ import annotations

import math
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

        adjacency = getattr(env.observation_space, "location_adjacency", ())
        assert adjacency, "Sokoban structural validation requires extracted move-dir adjacency"

        neighbors = agent.get_neighborhood(state)
        weights = agent.compute_kernel_weights(state, neighbors)
        assert len(neighbors) > 0
        assert np.all(np.isfinite(weights))
        assert np.isclose(float(np.sum(weights)), 1.0)
        center = env.observation_space.from_index(state)
        for idx in neighbors:
            candidate = env.observation_space.from_index(int(idx))
            assert env.observation_space.contains(candidate)
            assert np.array_equal(candidate[1:], center[1:])
            assert agent.get_distance(state, int(idx)) <= float(agent.search_radius) + 1e-9

        if len(center) > 1:
            changed_stone = center.copy()
            changed_stone[1] = (int(changed_stone[1]) + 1) % env.observation_space.nvec[1]
            assert math.isinf(agent.get_distance(state, env.observation_space.to_index(changed_stone)))

        for _ in range(5):
            action = env.action_space.sample(rng)
            obs, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(float(reward), float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                obs, _ = env.reset(seed=int(rng.integers(100000)))

        for _ in range(50):
            action = env.action_space.sample(rng)
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break
        print("PDDLGym Sokoban structural SBQ validation passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
