from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.trainer import build_env


ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _state(env, obs) -> dict[str, int]:
    return {name: int(value) for name, value in zip(env.feature_names, obs)}


def _slot_has(state: dict[str, int], max_slots: int, obj_type: int, color: int) -> bool:
    for idx in range(max_slots):
        if state[f"object_{idx}_type"] == obj_type and state[f"object_{idx}_color"] == color:
            return True
    return False


@pytest.mark.parametrize(
    ("package", "config_path", "action"),
    [
        ("minigrid", "applications/minigrid_fetch/config_qlearning.yaml", 0),
        ("minigrid", "applications/minigrid_putnear/config_qlearning.yaml", 0),
        ("minigrid", "applications/minigrid_blocked_unlock_pickup/config_qlearning.yaml", 0),
        ("boltcrypt", "applications/boltcrypt/config_qlearning.yaml", 0),
        ("connect_four_gymnasium", "applications/connect4/config_qlearning.yaml", 0),
        ("icu_sepsis", "applications/icu_sepsis/config_qlearning.yaml", 0),
    ],
)
def test_application_reset_step_and_render_semantics(package: str, config_path: str, action: int) -> None:
    pytest.importorskip(package)
    config = _load_config(config_path)
    if package == "boltcrypt":
        config["env"]["max_episode_steps"] = 5
    env = build_env(config)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(next_obs)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        frame = np.asarray(env.render(mode="rgb_array"))
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        if package == "boltcrypt":
            assert frame.shape[0] >= 160 and frame.shape[1] >= 160
    finally:
        env.close()


def test_fetch_mission_parser_variants_and_initial_targets() -> None:
    pytest.importorskip("minigrid")
    config = _load_config("applications/minigrid_fetch/config_qlearning.yaml")
    env = build_env(config)
    try:
        variants = [
            "get a red ball",
            "go get a blue key",
            "fetch a yellow ball",
            "go fetch a green key",
            "you must fetch a purple ball",
        ]
        for mission in variants:
            obj_type, color = env._target_from_mission(mission)
            assert obj_type != env.no_object_idx
            assert color != env.no_color_idx
        for seed in range(25):
            obs, _ = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            values = _state(env, obs)
            assert values["target_type"] != env.no_object_idx
            assert values["target_color"] != env.no_color_idx
            assert _slot_has(values, env.max_objects, values["target_type"], values["target_color"])
    finally:
        env.close()


def test_putnear_mission_parser_and_initial_targets() -> None:
    pytest.importorskip("minigrid")
    config = _load_config("applications/minigrid_putnear/config_qlearning.yaml")
    env = build_env(config)
    try:
        move_type, move_color, target_type, target_color = env._task_from_mission("put the yellow ball near the purple box")
        assert move_type != env.no_object_idx
        assert move_color != env.no_color_idx
        assert target_type != env.no_object_idx
        assert target_color != env.no_color_idx
        for seed in range(25):
            obs, _ = env.reset(seed=seed)
            assert env.observation_space.contains(obs)
            values = _state(env, obs)
            assert values["move_type"] != env.no_object_idx
            assert values["move_color"] != env.no_color_idx
            assert values["target_type"] != env.no_object_idx
            assert values["target_color"] != env.no_color_idx
            assert _slot_has(values, env.max_objects, values["move_type"], values["move_color"])
            assert _slot_has(values, env.max_objects, values["target_type"], values["target_color"])
    finally:
        env.close()


def test_putnear_carried_move_object_is_explicit() -> None:
    pytest.importorskip("minigrid")
    config = _load_config("applications/minigrid_putnear/config_qlearning.yaml")
    env = build_env(config)
    try:
        obs, info = env.reset(seed=0)
        values = _state(env, obs)
        move_obj = None
        move_pos = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.env.unwrapped.grid.get(x, y)
                if obj is None:
                    continue
                if env._type_idx(getattr(obj, "type", None)) == values["move_type"] and env._color_idx(getattr(obj, "color", None)) == values["move_color"]:
                    move_obj = obj
                    move_pos = (x, y)
                    break
            if move_obj is not None:
                break
        assert move_obj is not None and move_pos is not None
        env.env.unwrapped.grid.set(move_pos[0], move_pos[1], None)
        env.env.unwrapped.carrying = move_obj
        carried_obs = env._convert_obs({"mission": info["mission"]})
        carried = _state(env, carried_obs)
        assert carried["move_object_carried"] == 1
        assert _slot_has(carried, env.max_objects, carried["move_type"], carried["move_color"])
    finally:
        env.close()


def test_blocked_unlock_pickup_door_state_matches_minigrid_encoding() -> None:
    pytest.importorskip("minigrid")
    config = _load_config("applications/minigrid_blocked_unlock_pickup/config_qlearning.yaml")
    env = build_env(config)
    try:
        obs, _ = env.reset(seed=0)
        values = _state(env, obs)
        door = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.env.unwrapped.grid.get(x, y)
                if getattr(obj, "type", None) == "door":
                    door = obj
                    break
            if door is not None:
                break
        assert door is not None
        if door.is_open:
            expected = 0
        elif door.is_locked:
            expected = 2
        else:
            expected = 1
        assert values["door_state"] == expected
    finally:
        env.close()


def test_boltcrypt_local_time_limit_and_global_encoding() -> None:
    pytest.importorskip("boltcrypt")
    config = _load_config("applications/boltcrypt/config_qlearning.yaml")
    config["env"]["max_episode_steps"] = 1
    env = build_env(config)
    try:
        obs, _ = env.reset(seed=0)
        values = _state(env, obs)
        assert "agent_global_x" in values
        assert "agent_global_y" in values
        _, _, terminated, truncated, info = env.step(0)
        assert truncated
        assert info["TimeLimit.truncated"] == (not terminated)
    finally:
        env.close()


def test_connect4_masked_q_learning_avoids_full_columns() -> None:
    pytest.importorskip("connect_four_gymnasium")
    from applications.connect4.agent import Connect4MaskedQLearningAgent
    from applications.connect4.env import Connect4Env

    config = _load_config("applications/connect4/config_qlearning.yaml")
    env = Connect4Env(config)
    try:
        obs, info = env.reset(seed=0)
        assert info["valid_actions"] == list(range(7))
        board = np.zeros((6, 7), dtype=np.int64)
        board[:, 0] = 1
        full_col_obs = board.reshape(-1)
        assert env.observation_space.contains(full_col_obs)
        agent = Connect4MaskedQLearningAgent(env.observation_space, env.action_space, np.random.default_rng(0), epsilon=1.0)
        sampled = {agent.act(full_col_obs, explore=True) for _ in range(100)}
        assert 0 not in sampled
        agent.epsilon = 0.0
        greedy = {agent.act(full_col_obs, explore=False) for _ in range(100)}
        assert 0 not in greedy
    finally:
        env.close()


def test_icu_sepsis_action_and_terminal_semantics() -> None:
    pytest.importorskip("icu_sepsis")
    config = _load_config("applications/icu_sepsis/config_qlearning.yaml")
    env = build_env(config)
    try:
        assert env.observation_space.nvec == [716]
        assert env.action_space.n == 25
        for action in [0, 7, 24]:
            obs, _ = env.reset(seed=action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(next_obs)
            assert info["action_dose_levels"] == {"iv_fluid": action // 5, "vasopressor": action % 5}
        death = env._make_info({}, np.array([713], dtype=np.int64), action=0, reward=0.0, terminated=True)
        survival = env._make_info({}, np.array([714], dtype=np.int64), action=0, reward=1.0, terminated=True)
        other = env._make_info({}, np.array([10], dtype=np.int64), action=0, reward=0.0, terminated=True)
        assert death["terminal_outcome"] == "death"
        assert survival["terminal_outcome"] == "survival"
        assert other["terminal_outcome"] == "terminal"
    finally:
        env.close()
