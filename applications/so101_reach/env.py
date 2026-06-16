from __future__ import annotations

from typing import Any

import numpy as np

from applications.so101_reach.action_space import JointStepActionMapper
from applications.so101_reach.discretization import StateDiscretizer
from core.env_base import BaseEnv


class IsaacSo101ReachDiscreteEnv(BaseEnv):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        env_cfg = config.get("env", {})
        action_cfg = config.get("action", {})
        reward_cfg = config.get("reward", {})

        self.task = str(env_cfg.get("task", "Isaac-SO-ARM101-Reach-v0"))
        self.num_envs = int(env_cfg.get("num_envs", 1))
        if self.num_envs != 1:
            raise ValueError("BroadcastingQ SO101 reach currently supports num_envs=1")
        self.device = env_cfg.get("device")
        self.use_fabric = bool(env_cfg.get("use_fabric", True))
        self.render_mode = env_cfg.get("render_mode")
        self.ee_body_name = str(env_cfg.get("ee_body_name", "gripper_link"))
        self.command_name = str(env_cfg.get("command_name", "ee_pose"))
        self.command_resampling_time = env_cfg.get("command_resampling_time", "episode")
        self.success_threshold = float(env_cfg.get("success_threshold", 0.03))
        self.success_bonus = float(reward_cfg.get("success_bonus", env_cfg.get("success_bonus", 0.0)))
        self.terminate_on_success = bool(env_cfg.get("terminate_on_success", True))
        self.disable_observation_noise = bool(env_cfg.get("disable_observation_noise", True))
        self.use_shaped_reward = bool(reward_cfg.get("use_shaped_reward", False))
        self.raw_reward_scale = float(reward_cfg.get("raw_reward_scale", 1.0))
        self.progress_scale = float(reward_cfg.get("progress_scale", 0.0))
        self.distance_scale = float(reward_cfg.get("distance_scale", 0.0))
        self.failure_step_penalty = float(reward_cfg.get("failure_step_penalty", 0.0))

        self.joint_names = list(action_cfg.get(
            "joint_names",
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        ))
        self.action_mapper = JointStepActionMapper(
            joint_names=self.joint_names,
            step_size=float(action_cfg.get("step_size", 0.05)),
            include_noop=bool(action_cfg.get("include_noop", True)),
        )
        self.action_space = self.action_mapper.action_space
        self.discretizer = StateDiscretizer.from_config(config, self.joint_names)
        self.observation_space = self.discretizer.observation_space

        self._torch = None
        self._combine_frame_transforms = None
        self.env = self._make_env()
        self._robot = None
        self._joint_ids = None
        self._ee_body_id = None
        self._previous_distance = None
        self._previous_command_counter = None

    @property
    def action_names(self) -> list[str]:
        return self.action_mapper.action_names

    def reset(self, seed: int | None = None):
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.reset()
        info = self._get_info()
        self._previous_distance = float(info["distance_to_target"])
        self._previous_command_counter = self._get_command_counter()
        return self._get_discrete_obs(), info

    def step(self, action: int):
        continuous_action = self.action_mapper.to_continuous(action)
        torch = self._torch
        actions = torch.as_tensor(continuous_action, dtype=torch.float32, device=self.env.unwrapped.device).view(1, -1)
        _, reward, terminated, truncated, extras = self.env.step(actions)

        raw_reward = _first_float(reward)
        terminated_value = _first_bool(terminated)
        truncated_value = _first_bool(truncated)
        info = self._get_info()
        info["action_name"] = self.action_names[int(action)]
        info["continuous_action"] = continuous_action.tolist()
        info["raw_reward"] = raw_reward

        reward_value = self.raw_reward_scale * raw_reward
        current_distance = float(info["distance_to_target"])
        current_command_counter = self._get_command_counter()
        command_changed = (
            self._previous_command_counter is not None
            and current_command_counter != self._previous_command_counter
        )
        previous_distance = current_distance if self._previous_distance is None else float(self._previous_distance)
        progress_reward = 0.0 if command_changed else self.progress_scale * (previous_distance - current_distance)
        distance_reward = -self.distance_scale * current_distance
        if self.use_shaped_reward:
            reward_value += progress_reward + distance_reward + self.failure_step_penalty
        info["progress_reward"] = progress_reward
        info["distance_reward"] = distance_reward
        info["command_changed"] = command_changed

        if info["success"]:
            reward_value += self.success_bonus
            if self.terminate_on_success:
                terminated_value = True
        if isinstance(extras, dict) and "time_outs" in extras:
            info["time_out"] = _first_bool(extras["time_outs"])
        self._previous_distance = current_distance
        self._previous_command_counter = current_command_counter

        return self._get_discrete_obs(), reward_value, terminated_value, truncated_value, info

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise ValueError("IsaacSo101ReachDiscreteEnv supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Isaac returned no frame; construct env with render_mode='rgb_array'")
        return np.asarray(frame)

    def close(self) -> None:
        self.env.close()

    def _make_env(self):
        import gymnasium as gym
        import torch
        from isaaclab.utils.math import combine_frame_transforms
        import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
        import isaac_so_arm101.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self._torch = torch
        self._combine_frame_transforms = combine_frame_transforms

        cfg = parse_env_cfg(self.task, device=self.device, num_envs=self.num_envs, use_fabric=self.use_fabric)
        if self.disable_observation_noise and hasattr(cfg.observations, "policy"):
            cfg.observations.policy.enable_corruption = False
        if self.command_resampling_time is not None:
            if str(self.command_resampling_time).lower() == "episode":
                command_resampling_time = float(cfg.episode_length_s)
            else:
                command_resampling_time = float(self.command_resampling_time)
            cfg.commands.ee_pose.resampling_time_range = (command_resampling_time, command_resampling_time)
        cfg.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=self.joint_names,
            scale=1.0,
            use_zero_offset=True,
        )
        return gym.make(self.task, cfg=cfg, render_mode=self.render_mode)

    def _ensure_handles(self) -> None:
        if self._robot is not None:
            return
        self._robot = self.env.unwrapped.scene["robot"]
        self._joint_ids, _ = self._robot.find_joints(self.joint_names, preserve_order=True)
        body_ids, _ = self._robot.find_bodies(self.ee_body_name, preserve_order=True)
        self._ee_body_id = int(body_ids[0])

    def _continuous_state(self) -> tuple[np.ndarray, np.ndarray, float]:
        self._ensure_handles()
        robot = self._robot
        command = self.env.unwrapped.command_manager.get_command(self.command_name)
        target_pos_b = command[:, :3]
        target_pos_w, _ = self._combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_id]
        target_error = target_pos_w - ee_pos_w
        joint_pos = robot.data.joint_pos[:, self._joint_ids]
        distance = self._torch.norm(target_error, dim=1)
        return (
            target_error[0].detach().cpu().numpy(),
            joint_pos[0].detach().cpu().numpy(),
            float(distance[0].detach().cpu().item()),
        )

    def _get_command_counter(self) -> int:
        command_term = self.env.unwrapped.command_manager.get_term(self.command_name)
        return int(command_term.command_counter.reshape(-1)[0].detach().cpu().item())

    def _get_discrete_obs(self) -> np.ndarray:
        target_error, joint_pos, _ = self._continuous_state()
        return self.discretizer.encode(target_error, joint_pos)

    def _get_info(self) -> dict[str, Any]:
        target_error, joint_pos, distance = self._continuous_state()
        return {
            "target_error_xyz": target_error.tolist(),
            "joint_positions": joint_pos.tolist(),
            "distance_to_target": distance,
            "success": bool(distance <= self.success_threshold),
        }


def _first_float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(np.asarray(value).reshape(-1)[0])


def _first_bool(value) -> bool:
    if hasattr(value, "detach"):
        return bool(value.reshape(-1)[0].detach().cpu().item())
    return bool(np.asarray(value).reshape(-1)[0])

