"""Local smoke check for the Acrobot application only.

Run from the BroadcastingQ repository root:

    python -m applications.acrobot.smoke_check

This script validates the wrapper contract without editing core or agent code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from applications.acrobot.env import AcrobotEnv


def main() -> None:
    config_path = Path(__file__).with_name("config_qlearning.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    env = AcrobotEnv(config)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert obs.shape == (6,)
        assert env.action_space.n == 3
        assert info["success"] is False

        rng = np.random.default_rng(0)
        for _ in range(10):
            action = env.action_space.sample(rng)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            frame = env.render(mode="rgb_array")
            assert frame.ndim == 3 and frame.shape[-1] == 3
            if terminated or truncated:
                obs, info = env.reset(seed=0)
                assert env.observation_space.contains(obs)

        print("Acrobot wrapper smoke check passed.")
        print(f"observation_space.nvec = {env.observation_space.nvec}")
        print(f"observation_space.size = {env.observation_space.size}")
        print(f"observation_space.flat_dim = {env.observation_space.flat_dim}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
