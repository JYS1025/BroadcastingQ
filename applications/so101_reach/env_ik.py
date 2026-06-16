from __future__ import annotations

from typing import Any

import numpy as np

from applications.so101_reach.action_space import CartesianStepActionMapper
from applications.so101_reach.discretization import StateDiscretizer
from core.env_base import BaseEnv


class IsaacSo101ReachIKDiscreteEnv(BaseEnv):
    """SO101 reach wrapper with discrete task-space actions and Isaac Lab differential IK.

    Observation is binned target-relative end-effector xyz error. Actions are
    small Cartesian end-effector deltas; Isaac Lab maps those deltas to joint
    targets using DifferentialInverseKinematicsAction.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        env_cfg = config.get("env", {})
        action_cfg = config.get("action", {})
        reward_cfg = config.get("reward", {})

        self.task = str(env_cfg.get("task", "Isaac-SO-ARM101-Reach-v0"))
        self.num_envs = int(env_cfg.get("num_envs", 1))
        if self.num_envs != 1:
            raise ValueError("BroadcastingQ SO101 IK reach currently supports num_envs=1")
        self.device = env_cfg.get("device")
        self.use_fabric = bool(env_cfg.get("use_fabric", True))
        self.render_mode = env_cfg.get("render_mode")
        self.ee_body_name = str(env_cfg.get("ee_body_name", "gripper_link"))
        self.command_name = str(env_cfg.get("command_name", "ee_pose"))
        self.command_resampling_time = env_cfg.get("command_resampling_time", "episode")
        self.target_ranges = dict(env_cfg.get("target_ranges", {}))
        self.ee_relative_target_cfg = dict(env_cfg.get("ee_relative_target", {}))
        self.use_ee_relative_target = bool(self.ee_relative_target_cfg.get("enabled", False))
        self.ee_relative_target_radius = float(self.ee_relative_target_cfg.get("radius", 0.08))
        self.ee_relative_target_min_radius = float(self.ee_relative_target_cfg.get("min_radius", 0.0))
        if self.ee_relative_target_radius <= 0.0:
            raise ValueError("env.ee_relative_target.radius must be positive")
        if not 0.0 <= self.ee_relative_target_min_radius <= self.ee_relative_target_radius:
            raise ValueError("env.ee_relative_target.min_radius must be in [0, radius]")
        self.success_threshold = float(env_cfg.get("success_threshold", 0.03))
        self.success_bonus = float(reward_cfg.get("success_bonus", env_cfg.get("success_bonus", 0.0)))
        self.terminate_on_success = bool(env_cfg.get("terminate_on_success", True))
        self.disable_observation_noise = bool(env_cfg.get("disable_observation_noise", True))
        max_episode_steps = env_cfg.get("max_episode_steps")
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError("env.max_episode_steps must be positive when set")
        # Hold-position steps after reset so the arm settles under gravity before the
        # episode (and before the EE-relative target is sampled). Removes the post-reset
        # sag transient that otherwise biases the z error and the shaped reward.
        self.reset_settle_steps = int(env_cfg.get("reset_settle_steps", 0))
        if self.reset_settle_steps < 0:
            raise ValueError("env.reset_settle_steps must be non-negative")
        # Reset posture override + (optional) tighter reset scale. The task resets joints by
        # MULTIPLYING the default pose, so a default of 0 (shoulder_lift, elbow_flex) can never
        # become non-zero -> the arm always starts horizontal. These let us start it crouched.
        self.reset_joint_pos = dict(env_cfg.get("reset_joint_pos", {}))
        self.reset_joint_pos_range = env_cfg.get("reset_joint_pos_range")
        # Arm actuator authority overrides (task-local). Raise to fix gravity sag / weak
        # vertical reach when the probe shows torque starvation (no joint at a limit).
        self.arm_effort_limit = env_cfg.get("arm_effort_limit")
        self.arm_stiffness_scale = env_cfg.get("arm_stiffness_scale")
        self.use_shaped_reward = bool(reward_cfg.get("use_shaped_reward", False))
        self.raw_reward_scale = float(reward_cfg.get("raw_reward_scale", 1.0))
        self.progress_scale = float(reward_cfg.get("progress_scale", 0.0))
        self.distance_scale = float(reward_cfg.get("distance_scale", 0.0))
        self.failure_step_penalty = float(reward_cfg.get("failure_step_penalty", 0.0))

        self.ik_joint_names = list(action_cfg.get(
            "ik_joint_names",
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        ))
        self.ik_method = str(action_cfg.get("ik_method", "dls"))
        self.ik_params = action_cfg.get("ik_params")
        # "absolute_base": command an absolute target = current_ee_pos_b + delta in the
        # robot base frame, matching the frame the target-error observation is measured in.
        # This avoids the relative-mode ambiguity where the IK delta is applied in the
        # end-effector frame and gets rotated off the observation axes.
        self.ik_command_mode = str(action_cfg.get("ik_command_mode", "absolute_base")).lower()
        if self.ik_command_mode not in ("absolute_base", "relative"):
            raise ValueError("action.ik_command_mode must be 'absolute_base' or 'relative'")
        self.base_step_size = float(action_cfg.get("step_size", 0.02))
        # Hold each commanded target for this many sim steps before the next agent
        # decision. One sim step is not enough for the joint controller to track a
        # commanded Cartesian target (it under-shoots, and in z gravity wins), so a
        # single agent action barely moves the EE. Repeating lets the arm converge.
        self.control_substeps = int(action_cfg.get("control_substeps", 1))
        if self.control_substeps < 1:
            raise ValueError("action.control_substeps must be >= 1")
        # Anti-windup bound (m) for the persistent absolute setpoint: how far the commanded
        # point may lead the actual EE. Keeps a standing position error so the PD resists
        # gravity (no re-anchoring sag), while stopping the setpoint from running away on
        # axes the arm cannot reach. 0 disables the clamp (pure integration).
        self.command_max_lag = float(action_cfg.get("command_max_lag", 0.08))
        self.adaptive_step_size = _parse_adaptive_step_size(action_cfg.get("adaptive_step_size"))
        self.action_mapper = CartesianStepActionMapper(
            step_size=self.base_step_size,
            include_noop=bool(action_cfg.get("include_noop", True)),
        )
        self.action_space = self.action_mapper.action_space
        self.discretizer = StateDiscretizer.from_config(config, [])
        self.observation_space = self.discretizer.observation_space

        self._torch = None
        self._combine_frame_transforms = None
        self._subtract_frame_transforms = None
        self.env = self._make_env()
        self._robot = None
        self._ee_body_id = None
        self._previous_distance = None
        self._previous_command_counter = None
        self._episode_step_count = 0
        # Persistent absolute base-frame setpoint, integrated by the agent's deltas. Anchored
        # to the EE at reset; NOT re-read from the (sagging) EE each step. See step().
        self._cmd_pos_b = None

    @property
    def action_names(self) -> list[str]:
        return self.action_mapper.action_names

    def reset(self, seed: int | None = None):
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.reset()
        self._episode_step_count = 0
        self._settle()
        if self.use_ee_relative_target:
            self._resample_ee_relative_target()
        # Anchor the persistent setpoint to the settled EE pose (after settle, before the
        # episode). From here it only moves by the agent's commanded deltas.
        if self.ik_command_mode != "relative":
            self._cmd_pos_b = self._ee_pos_b().clone()
        info = self._get_info()
        self._previous_distance = float(info["distance_to_target"])
        self._previous_command_counter = self._get_command_counter()
        return self._get_discrete_obs(), info

    def step(self, action: int):
        pre_info = self._get_info()
        action_step_size = self._step_size_for_distance(float(pre_info["distance_to_target"]))
        continuous_action = self.action_mapper.to_continuous(action)
        if np.any(continuous_action):
            continuous_action = np.sign(continuous_action) * action_step_size
        torch = self._torch
        delta = torch.as_tensor(continuous_action, dtype=torch.float32, device=self.env.unwrapped.device).view(1, -1)
        if self.ik_command_mode == "relative":
            target = delta
            hold = torch.zeros_like(delta)
        else:
            # Absolute target in the base frame: a +x delta moves the EE +x in the same
            # frame target_error is measured, so the action/observation axes stay aligned.
            #
            # The setpoint is INTEGRATED from a fixed reference (self._cmd_pos_b += delta),
            # NOT rebuilt from the live EE each step. Re-reading the EE (`_ee_pos_b() + delta`)
            # silently re-commanded "hold current height" every step, so gravity ratcheted the
            # EE down ~0.085 m/episode even under noop. A setpoint fixed in space keeps a
            # standing position error the PD resists, so the arm holds height. The clamp is
            # anti-windup: it bounds how far the setpoint may lead the EE so it cannot run
            # away on axes the arm cannot reach (e.g. +z).
            if self._cmd_pos_b is None:
                self._cmd_pos_b = self._ee_pos_b().clone()
            self._cmd_pos_b = self._cmd_pos_b + delta
            if self.command_max_lag > 0.0:
                ee = self._ee_pos_b()
                self._cmd_pos_b = torch.clamp(
                    self._cmd_pos_b, ee - self.command_max_lag, ee + self.command_max_lag
                )
            target = self._cmd_pos_b
            hold = target

        # Apply the command, then hold the same target for the remaining sub-steps so the
        # arm converges to it (absolute mode holds the fixed point; relative mode applies
        # the delta once then commands zero motion).
        raw_reward = 0.0
        terminated_value = False
        truncated_value = False
        extras: Any = {}
        for substep in range(self.control_substeps):
            cmd = target if substep == 0 else hold
            _, reward, terminated, truncated, extras = self.env.step(cmd)
            raw_reward += _first_float(reward)
            terminated_value = _first_bool(terminated)
            truncated_value = _first_bool(truncated)
            if terminated_value or truncated_value:
                break
        info = self._get_info()
        info["action_name"] = self.action_names[int(action)]
        info["continuous_action"] = continuous_action.tolist()
        info["action_step_size"] = action_step_size
        info["pre_step_distance_to_target"] = float(pre_info["distance_to_target"])
        info["raw_reward"] = raw_reward
        self._episode_step_count += 1
        info["episode_step"] = self._episode_step_count

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
        wrapper_time_limit = (
            self.max_episode_steps is not None
            and self._episode_step_count >= self.max_episode_steps
            and not terminated_value
        )
        if wrapper_time_limit:
            truncated_value = True
        info["wrapper_time_limit"] = wrapper_time_limit
        if isinstance(extras, dict) and "time_outs" in extras:
            info["time_out"] = _first_bool(extras["time_outs"])
        self._previous_distance = current_distance
        self._previous_command_counter = current_command_counter

        return self._get_discrete_obs(), reward_value, terminated_value, truncated_value, info

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise ValueError("IsaacSo101ReachIKDiscreteEnv supports only mode='rgb_array'")
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Isaac returned no frame; construct env with render_mode='rgb_array'")
        return np.asarray(frame)

    def close(self) -> None:
        self.env.close()

    def _make_env(self):
        import gymnasium as gym
        import torch
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
        import isaac_so_arm101.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        self._torch = torch
        self._combine_frame_transforms = combine_frame_transforms
        self._subtract_frame_transforms = subtract_frame_transforms

        cfg = parse_env_cfg(self.task, device=self.device, num_envs=self.num_envs, use_fabric=self.use_fabric)
        if self.disable_observation_noise and hasattr(cfg.observations, "policy"):
            cfg.observations.policy.enable_corruption = False
        # Crouched reset posture. The reach task's reset event (reset_joints_by_scale)
        # MULTIPLIES the default joint pos, so joints whose default is 0 (shoulder_lift,
        # elbow_flex) are pinned at 0 forever -> the arm always starts horizontal, the worst
        # pose for gravity (max torque, near-singular for vertical motion). Override the init
        # defaults so the arm starts self-supported with vertical-reach headroom.
        if self.reset_joint_pos and hasattr(cfg.scene, "robot"):
            jp = dict(cfg.scene.robot.init_state.joint_pos)
            jp.update({str(k): float(v) for k, v in self.reset_joint_pos.items()})
            cfg.scene.robot.init_state.joint_pos = jp
        # Optionally tighten the multiplicative reset scale so the start pose is consistent
        # (range is a fraction of the default pose: (1,1) = exactly the default, no spread).
        if self.reset_joint_pos_range is not None and hasattr(cfg, "events"):
            reset_term = getattr(cfg.events, "reset_robot_joints", None)
            if reset_term is not None:
                reset_term.params["position_range"] = (
                    float(self.reset_joint_pos_range[0]),
                    float(self.reset_joint_pos_range[1]),
                )
        # Actuator authority override (task-local): the default 1.9 Nm effort limit leaves the
        # arm torque-starved (cannot hold pose or lift in z). Raise effort / stiffness here
        # without editing the shared robot asset (which other tasks share).
        if (self.arm_effort_limit is not None or self.arm_stiffness_scale is not None) and hasattr(cfg.scene, "robot"):
            arm_act = cfg.scene.robot.actuators["arm"]
            if self.arm_effort_limit is not None:
                arm_act.effort_limit_sim = float(self.arm_effort_limit)
            if self.arm_stiffness_scale is not None:
                scale = float(self.arm_stiffness_scale)
                arm_act.stiffness = _scale_gain(arm_act.stiffness, scale)
                arm_act.damping = _scale_gain(arm_act.damping, scale)
        # Each agent step now runs control_substeps physics steps, which consumes the
        # simulator's episode_length_s (seconds) budget that many times faster. Without
        # scaling it, the underlying env times out (and resamples the command target to
        # the default ranges) long before our max_episode_steps cap. Scale it so the
        # wrapper cap stays the binding limit and the EE-relative target is not resampled
        # mid-episode.
        if self.control_substeps > 1:
            cfg.episode_length_s = float(cfg.episode_length_s) * self.control_substeps
        if self.command_resampling_time is not None:
            if str(self.command_resampling_time).lower() == "episode":
                command_resampling_time = float(cfg.episode_length_s)
            else:
                command_resampling_time = float(self.command_resampling_time)
            cfg.commands.ee_pose.resampling_time_range = (command_resampling_time, command_resampling_time)
        if self.target_ranges:
            command_ranges = cfg.commands.ee_pose.ranges
            for key in ("pos_x", "pos_y", "pos_z"):
                if key in self.target_ranges:
                    setattr(command_ranges, key, _as_range_pair(self.target_ranges[key], key))
        cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=self.ik_joint_names,
            body_name=self.ee_body_name,
            controller=DifferentialIKControllerCfg(
                command_type="position",
                use_relative_mode=(self.ik_command_mode == "relative"),
                ik_method=self.ik_method,
                ik_params=self.ik_params,
            ),
            scale=1.0,
        )
        return gym.make(self.task, cfg=cfg, render_mode=self.render_mode)

    def _ensure_handles(self) -> None:
        if self._robot is not None:
            return
        self._robot = self.env.unwrapped.scene["robot"]
        body_ids, _ = self._robot.find_bodies(self.ee_body_name, preserve_order=True)
        self._ee_body_id = int(body_ids[0])

    def _continuous_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        self._ensure_handles()
        robot = self._robot
        command = self.env.unwrapped.command_manager.get_command(self.command_name)
        target_pos_b = command[:, :3]
        target_pos_w, _ = self._combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_id]
        ee_pos_b, _ = self._subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
            robot.data.body_quat_w[:, self._ee_body_id],
        )
        target_error_b = target_pos_b - ee_pos_b
        distance = self._torch.norm(target_error_b, dim=1)
        return (
            target_error_b[0].detach().cpu().numpy(),
            ee_pos_w[0].detach().cpu().numpy(),
            target_pos_w[0].detach().cpu().numpy(),
            float(distance[0].detach().cpu().item()),
        )

    def _settle(self) -> None:
        if self.reset_settle_steps <= 0:
            return
        self._ensure_handles()
        torch = self._torch
        device = self.env.unwrapped.device
        # Hold a FIXED pose during settle (captured once), consistent with the persistent
        # setpoint used in step(). Re-reading the EE each step would let it sag away during
        # the settle window instead of converging to a held pose.
        hold_fixed = None if self.ik_command_mode == "relative" else self._ee_pos_b().clone()
        for _ in range(self.reset_settle_steps):
            if self.ik_command_mode == "relative":
                hold = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
            else:
                hold = hold_fixed
            self.env.step(hold)

    def _ee_pos_b(self):
        """Current end-effector position in the robot base/root frame, shape (num_envs, 3).

        Uses the same root-relative transform as the observation, so an absolute IK
        command built from this is expressed in the observation's frame.
        """
        self._ensure_handles()
        robot = self._robot
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_id]
        ee_pos_b, _ = self._subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        return ee_pos_b

    def _get_command_counter(self) -> int:
        command_term = self.env.unwrapped.command_manager.get_term(self.command_name)
        return int(command_term.command_counter.reshape(-1)[0].detach().cpu().item())

    def _resample_ee_relative_target(self) -> None:
        self._ensure_handles()
        torch = self._torch
        robot = self._robot
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_id]
        ee_pos_b, _ = self._subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        direction = torch.randn((self.num_envs, 3), device=self.env.unwrapped.device)
        direction = direction / torch.clamp(torch.norm(direction, dim=1, keepdim=True), min=1.0e-6)
        min_r = self.ee_relative_target_min_radius
        max_r = self.ee_relative_target_radius
        radius = torch.rand((self.num_envs, 1), device=self.env.unwrapped.device)
        if min_r > 0.0:
            radius = (radius * (max_r ** 3 - min_r ** 3) + min_r ** 3) ** (1.0 / 3.0)
        else:
            radius = radius ** (1.0 / 3.0) * max_r
        command_term = self.env.unwrapped.command_manager.get_term(self.command_name)
        command_term.pose_command_b[:, :3] = ee_pos_b + direction * radius

    def _step_size_for_distance(self, distance: float) -> float:
        if not self.adaptive_step_size:
            return self.base_step_size
        for threshold, step_size in self.adaptive_step_size:
            if distance >= threshold:
                return step_size
        return self.base_step_size

    def _get_discrete_obs(self) -> np.ndarray:
        target_error, _, _, _ = self._continuous_state()
        return self.discretizer.encode(target_error, [])

    def _get_info(self) -> dict[str, Any]:
        target_error, ee_pos, target_pos, distance = self._continuous_state()
        return {
            "target_error_xyz": target_error.tolist(),
            "ee_position_xyz": ee_pos.tolist(),
            "target_position_xyz": target_pos.tolist(),
            "distance_to_target": distance,
            "success": bool(distance <= self.success_threshold),
        }


def _scale_gain(gain, scale: float):
    """Scale an actuator gain that may be a scalar or a per-joint dict."""
    if isinstance(gain, dict):
        return {k: float(v) * scale for k, v in gain.items()}
    return float(gain) * scale


def _as_range_pair(value, name: str) -> tuple[float, float]:
    out = tuple(float(v) for v in value)
    if len(out) != 2:
        raise ValueError(f"env.target_ranges.{name} must contain exactly two values")
    if out[0] > out[1]:
        raise ValueError(f"env.target_ranges.{name} low must be <= high")
    return out


def _parse_adaptive_step_size(config) -> list[tuple[float, float]]:
    if not config or not bool(config.get("enabled", False)):
        return []
    schedule = []
    for item in config.get("schedule", []):
        schedule.append((float(item["distance_gte"]), float(item["step_size"])))
    if not schedule:
        raise ValueError("action.adaptive_step_size.schedule must contain at least one item when enabled")
    if any(step_size <= 0.0 for _, step_size in schedule):
        raise ValueError("adaptive step sizes must be positive")
    return sorted(schedule, key=lambda item: item[0], reverse=True)


def _first_float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(np.asarray(value).reshape(-1)[0])


def _first_bool(value) -> bool:
    if hasattr(value, "detach"):
        return bool(value.reshape(-1)[0].detach().cpu().item())
    return bool(np.asarray(value).reshape(-1)[0])
