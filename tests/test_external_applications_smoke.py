from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.trainer import build_env
from core.utils import import_from_string


ROOT = Path(__file__).resolve().parents[1]

APP_CONFIGS = {
    "minigrid": [
        ROOT / "applications/minigrid_fetch/config_qlearning.yaml",
        ROOT / "applications/minigrid_fetch/config_dqn.yaml",
        ROOT / "applications/minigrid_fetch/config_sbq.yaml",
        ROOT / "applications/minigrid_putnear/config_qlearning.yaml",
        ROOT / "applications/minigrid_putnear/config_dqn.yaml",
        ROOT / "applications/minigrid_putnear/config_sbq.yaml",
        ROOT / "applications/minigrid_blocked_unlock_pickup/config_qlearning.yaml",
        ROOT / "applications/minigrid_blocked_unlock_pickup/config_dqn.yaml",
        ROOT / "applications/minigrid_blocked_unlock_pickup/config_sbq.yaml",
    ],
    "boltcrypt": [
        ROOT / "applications/boltcrypt/config_qlearning.yaml",
        ROOT / "applications/boltcrypt/config_dqn.yaml",
        ROOT / "applications/boltcrypt/config_sbq.yaml",
    ],
    "connect_four_gymnasium": [
        ROOT / "applications/connect4/config_qlearning.yaml",
        ROOT / "applications/connect4/config_dqn.yaml",
        ROOT / "applications/connect4/config_sbq.yaml",
    ],
    "icu_sepsis": [
        ROOT / "applications/icu_sepsis/config_qlearning.yaml",
        ROOT / "applications/icu_sepsis/config_dqn.yaml",
        ROOT / "applications/icu_sepsis/config_sbq.yaml",
    ],
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("config_path", [path for paths in APP_CONFIGS.values() for path in paths])
def test_external_application_config_entrypoints_import(config_path: Path) -> None:
    config = _load_config(config_path)
    import_from_string(config["application"]["entrypoint"])
    import_from_string(config["application"]["visualizer"])


@pytest.mark.parametrize(
    ("package", "config_path", "steps"),
    [
        ("minigrid", ROOT / "applications/minigrid_fetch/config_qlearning.yaml", 5),
        ("minigrid", ROOT / "applications/minigrid_putnear/config_qlearning.yaml", 5),
        ("minigrid", ROOT / "applications/minigrid_blocked_unlock_pickup/config_qlearning.yaml", 5),
        ("boltcrypt", ROOT / "applications/boltcrypt/config_qlearning.yaml", 5),
        ("connect_four_gymnasium", ROOT / "applications/connect4/config_qlearning.yaml", 10),
        ("icu_sepsis", ROOT / "applications/icu_sepsis/config_qlearning.yaml", 5),
    ],
)
def test_external_application_env_smoke(package: str, config_path: Path, steps: int) -> None:
    pytest.importorskip(package)
    config = _load_config(config_path)
    env = build_env(config)
    rng = np.random.default_rng(0)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        for step in range(steps):
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(float(reward), float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if terminated or truncated:
                obs, info = env.reset(seed=step + 1)
                assert env.observation_space.contains(obs)
        frame = env.render(mode="rgb_array")
        frame = np.asarray(frame)
        assert frame.ndim == 3
        assert frame.shape[2] in (3, 4)
        assert frame.shape[0] > 0 and frame.shape[1] > 0
    finally:
        env.close()

