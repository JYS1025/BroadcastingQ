from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from agents.broadcasting_q import SBQAgent
from applications.so101_reach.agents import EuclideanSBQAgent
from applications.so101_reach.env import IsaacSo101ReachDiscreteEnv
from applications.so101_reach.env_ik import IsaacSo101ReachIKDiscreteEnv
from core.agent_base import BaseAgent, Transition
from core.logging import CSVLogger
from core.schedules import make_schedule
from core.utils import ensure_dir

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def build_agent(config: dict[str, Any], env: Any, rng: np.random.Generator) -> BaseAgent:
    agent_cfg = dict(config.get("agent", {}))
    name = str(agent_cfg.pop("name")).lower()
    epsilon_cfg = agent_cfg.pop("epsilon", None)
    initial_epsilon = 0.0
    if epsilon_cfg is not None:
        initial_epsilon = float(epsilon_cfg.get("start", epsilon_cfg.get("value", 0.0)))

    common = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "rng": rng,
        "epsilon": initial_epsilon,
    }
    if name in {"q_learning", "qlearning"}:
        return QLearningAgent(**common, **agent_cfg)
    if name == "dqn":
        return DQNAgent(**common, **agent_cfg)
    if name == "sbq":
        distance_metric = str(agent_cfg.pop("distance_metric", "euclidean")).lower()
        if distance_metric == "hamming":
            return SBQAgent(**common, **agent_cfg)
        if distance_metric != "euclidean":
            raise ValueError(f"Unsupported SBQ distance_metric: {distance_metric}")
        max_neighbor_delta = int(agent_cfg.pop("max_neighbor_delta", 1))
        return EuclideanSBQAgent(
            **common,
            distance_centers=env.discretizer.distance_centers,
            max_neighbor_delta=max_neighbor_delta,
            **agent_cfg,
        )
    raise ValueError(f"Unsupported agent name: {name}")


def build_env(config: dict[str, Any]):
    env_type = str(config.get("env", {}).get("type", "joint")).lower()
    if env_type in {"joint", "joint_step"}:
        return IsaacSo101ReachDiscreteEnv(config)
    if env_type in {"ik", "ik_xyz", "task_space"}:
        return IsaacSo101ReachIKDiscreteEnv(config)
    raise ValueError(f"Unsupported SO101 reach env.type: {env_type}")


def train(config: dict[str, Any], simulation_app=None) -> Path:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    env = build_env(config)
    agent = build_agent(config, env, rng)
    epsilon_schedule = make_schedule(config.get("agent", {}).get("epsilon"), default=getattr(agent, "epsilon", 0.0))
    output_dir = _prepare_output_dir(config)
    _write_experiment_description(output_dir, env, config)

    training = config.get("training", {})
    total_steps = int(training.get("total_steps", 100_000))
    eval_interval = int(training.get("eval_interval", 5_000))
    eval_episodes = int(training.get("eval_episodes", 20))
    log_interval = int(training.get("log_interval", 1_000))
    save_interval = int(training.get("save_interval", 25_000))
    progress_bar = bool(training.get("progress_bar", True))

    metrics_logger = CSVLogger(
        output_dir / "metrics.csv",
        [
            "step",
            "episode",
            "episode_return",
            "episode_length",
            "success",
            "distance_to_target",
            "epsilon",
            "td_error",
            "loss",
            "q_value",
            "target",
            "td_error_mean",
            "q_mean",
        ],
    )
    eval_logger = CSVLogger(
        output_dir / "eval_metrics.csv",
        ["step", "eval_return_mean", "eval_return_std", "eval_length_mean", "eval_success_rate", "eval_distance_mean"],
    )

    obs, info = env.reset(seed=seed)
    episode = 0
    episode_return = 0.0
    episode_length = 0
    last_update: dict[str, Any] = {}

    steps = range(1, total_steps + 1)
    if tqdm is not None and progress_bar:
        steps = tqdm(steps, desc=f"so101 {config['agent']['name']}", dynamic_ncols=True)

    try:
        for step in steps:
            if simulation_app is not None and not simulation_app.is_running():
                break
            epsilon = float(epsilon_schedule(step))
            if hasattr(agent, "epsilon"):
                agent.epsilon = epsilon

            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += float(reward)
            episode_length += 1
            transition = Transition(obs, int(action), reward, next_obs, terminated, truncated, info)
            last_update = agent.update(transition)
            obs = next_obs

            if log_interval > 0 and step % log_interval == 0:
                _log_train_row(metrics_logger, step, episode, episode_return, episode_length, info, epsilon, last_update)

            if terminated or truncated:
                _log_train_row(metrics_logger, step, episode, episode_return, episode_length, info, epsilon, last_update)
                episode += 1
                obs, info = env.reset()
                episode_return = 0.0
                episode_length = 0

            if eval_interval > 0 and step % eval_interval == 0:
                eval_logger.write(evaluate(env, agent, step, eval_episodes, seed + 10_000))
                obs, info = env.reset()
                episode_return = 0.0
                episode_length = 0

            if save_interval > 0 and step % save_interval == 0:
                _save_checkpoint(agent, config["agent"]["name"], output_dir, step)

        _save_checkpoint(agent, config["agent"]["name"], output_dir, step if "step" in locals() else 0)
    finally:
        metrics_logger.close()
        eval_logger.close()
        env.close()

    return output_dir


def evaluate(env: Any, agent: BaseAgent, step: int, episodes: int, seed: int) -> dict[str, float]:
    returns = []
    lengths = []
    successes = []
    distances = []
    max_steps = int(env.config.get("training", {}).get("eval_max_steps", 360))
    for episode_idx in range(int(episodes)):
        obs, info = env.reset(seed=seed + episode_idx)
        done = False
        episode_return = 0.0
        length = 0
        while not done and length < max_steps:
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_return += float(reward)
            length += 1
            done = bool(terminated or truncated)
        returns.append(episode_return)
        lengths.append(length)
        successes.append(float(info.get("success", False)))
        distances.append(float(info.get("distance_to_target", np.nan)))
    return {
        "step": step,
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_distance_mean": float(np.nanmean(distances)),
    }


def _prepare_output_dir(config: dict[str, Any]) -> Path:
    logging_cfg = config.get("logging", {})
    output_root = Path(logging_cfg.get("output_root", "outputs"))
    run_name = str(logging_cfg.get("run_name", f"so101_reach_{config['agent']['name']}_seed{config.get('seed', 0)}"))
    output_dir = ensure_dir(output_root / run_name)
    ensure_dir(output_dir / "checkpoints")
    return output_dir


def _write_experiment_description(output_dir: Path, env: Any, config: dict[str, Any]) -> None:
    lines = [
        f"agent: {config['agent']['name']}",
        f"task: {env.task}",
        f"observation_nvec: {env.observation_space.nvec}",
        f"action_count: {env.action_space.n}",
        f"action_names: {env.action_names}",
        "",
        "bins:",
    ]
    for item in env.discretizer.describe_bins():
        lines.append(f"- {item}")
    (output_dir / "experiment_description.txt").write_text("\n".join(lines), encoding="utf-8")


def _log_train_row(
    logger: CSVLogger,
    step: int,
    episode: int,
    episode_return: float,
    episode_length: int,
    info: dict[str, Any],
    epsilon: float,
    update_info: dict[str, Any],
) -> None:
    row = {
        "step": step,
        "episode": episode,
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "success": bool(info.get("success", False)),
        "distance_to_target": float(info.get("distance_to_target", np.nan)),
        "epsilon": float(epsilon),
    }
    row.update(update_info or {})
    logger.write(row)


def _save_checkpoint(agent: BaseAgent, agent_name: str, output_dir: Path, step: int) -> None:
    suffix = ".pt" if str(agent_name).lower() == "dqn" else ".pkl"
    agent.save(str(output_dir / "checkpoints" / f"agent_step_{step}{suffix}"))

