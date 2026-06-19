# SO101 IK SBQ Context

Last updated: 2026-06-10

## Project Scope

- Task: `Isaac-SO-ARM101-Reach-v0`
- Focus: make tabular/SBQ-style learning work on the IK/task-space SO101 reach wrapper.
- Main files:
  - `applications/so101_reach/config_ik_sbq.yaml`
  - `applications/so101_reach/env_ik.py`
  - `applications/so101_reach/agents.py` (`EuclideanSBQAgent`)
  - `agents/broadcasting_q.py` (`SBQAgent`)
  - `applications/so101_reach/runner.py`
  - `applications/so101_reach/probe_actions.py` (diagnostic tool added this session)

---

# TL;DR (2026-06-10 debugging session)

The reach was not learning (greedy success stuck ~5-10%, distance flat ~0.09 m). It turned
out to be **mostly an environment / robot-control problem, not an RL problem**. We found and
fixed a chain of issues, in this order:

1. **SBQ broadcasting too aggressive + sign-blind** → milder annealing + sign-aware neighborhood.
2. **Eval was disabled** → re-enabled greedy eval (the real signal; training rows use epsilon).
3. **IK action frame misaligned with the observation frame** → command absolute targets in the
   base frame (the single most important fix; actions now move the correct axes).
4. **One sim step per action under-tracked badly + gravity sag** → `control_substeps` (hold each
   commanded target for N sim steps); also `reset_settle_steps`.
5. **`wrist_flex` driven into its joint limit by redundant IK** → use a non-redundant 3-DOF IK
   joint set (`shoulder_pan, shoulder_lift, elbow_flex`).

After all five, the action probe is clean (x/y reach ~0.85-0.95 of commanded, no joints at
limit). z is the residual weak axis (~0.45-0.54 of commanded) due to arm geometry near the
vertical workspace boundary. Training rerun is pending under run name `so101_reach_ik_sbq_3dof_seed0`.

**Key lesson:** pure Q-learning ALSO failed, which (correctly) pointed at the environment, not
the agent. The `probe_actions.py` tool is what localized each env bug — use it before tuning RL.

---

# Debugging Session 2026-06-10 (problems, fixes, results)

### Problem 1 — SBQ broadcasting too aggressive and sign-blind
- **Symptom:** SBQ table noisy, no stable trend.
- **Diagnosis:**
  - The combined Q (`q_global + beta_t * q_kernel`) is the learned quantity, so `q_global` is
    entangled with the kernel. Convergence needs the kernel term to stabilize/anneal.
  - With `beta_0=2.0, p=0.25, alpha_k=2.0` the kernel was large and re-injected on every
    neighbor visit forever (beta decays very slowly), so the target never settled.
  - `EuclideanSBQAgent` (used because `distance_metric: euclidean`) broadcasts in scaled
    bin-center space with `search_radius=2.5`, which **crosses the zero-error bin** near the goal,
    sending a TD error learned where `-x` is optimal into states where `+x` is optimal.
- **Fix (config `agent:`):** `beta_0 2.0->1.0`, `p 0.25->0.5`, `alpha_k 2.0->0.3`,
  `search_radius 2.5->1.5`.
- **Fix (code `agents.py` `EuclideanSBQAgent`):** sign-aware `get_neighborhood`.
  - Added `self.center_bins = argmin(|distance_centers|)` per axis (the zero-error bin, index 5).
  - In `get_neighborhood`, keep only candidate bins on the **same side of the center bin** as the
    source (the center bin itself always allowed). Verified: source x-bin 7 -> neighbors {5,6,7,8};
    source x-bin 3 -> {2,3,4,5}; source x-bin 5 -> both sides.
- **Result:** necessary but NOT sufficient — still flat. (Real cause was downstream, in the env.)

### Problem 2 — Eval disabled, so we were reading exploratory noise
- **Symptom:** "success rate" judged from training rows, which run at epsilon>=0.05.
- **Fix (config):** `eval_interval 0->10000`, `eval_episodes 0->20`.
- **Fix (code `runner.py`):** uncommented the eval block in `train()` (calls `evaluate(...)`,
  which uses `agent.act(obs, explore=False)` = greedy). Results -> `eval_metrics.csv`.
- **Result:** greedy eval confirmed the policy was essentially random; `eval_distance_mean`
  flat ~0.09 m for 200k+ steps (worse than the 0.04-0.08 start), i.e. it drifted AWAY. This
  flatness (plus pure-Q also failing) is what redirected us from RL tuning to the environment.

### Problem 3 — IK action frame misaligned with observation frame (THE big one)
- **Tool:** added `probe_actions.py`. For each action it holds the action and reports the mean
  change in `target_error_xyz`. Expectation: `+x` => `d(err_x) < 0` and dominates.
- **Symptom:** probe showed mismatches — actions did not cleanly move the expected axis.
- **Diagnosis:** observation `target_error` is in the robot **base/root frame**, but the
  differential-IK **relative** position command is not guaranteed to be applied in that frame
  (it can be applied in the EE/body frame and get rotated off the observation axes).
- **Fix (code `env_ik.py`):** new `action.ik_command_mode` (default `absolute_base`).
  - `_make_env`: `use_relative_mode=(ik_command_mode == "relative")` (now false by default).
  - `step()`: command an **absolute** target `= _ee_pos_b() + delta` in the base frame.
  - Added `_ee_pos_b()` helper that computes EE position via the same
    `subtract_frame_transforms(root, ...)` the observation uses, so action and observation share
    a frame by construction.
- **Result:** after subtracting the noop baseline, all 6 actions moved the **correct axis in the
  correct direction** (x/y clean, no cross-talk on y). Frame bug fixed. But training still flat —
  revealed Problem 4.

### Problem 4 — Severe per-step under-tracking + gravity sag
- **Symptom:** probe showed every action (incl. `noop`) had a large common-mode `+z` drift
  (~0.011 over 5 steps); commanding `+z` still net-fell. Per-step EE motion was a small fraction
  of the commanded step. Training `|td_error|` stuck ~0.72 (value function never settling),
  median distance flat at the start value.
- **Diagnosis:** one agent action = one sim step is not enough time for the joint controller to
  track the commanded Cartesian target; in z, gravity wins, so the EE keeps falling out of reach.
- **Fix (code `env_ik.py`):**
  - `action.control_substeps` (default 1): hold each commanded target for N sim steps so the arm
    converges to it (absolute mode holds the fixed point; relative applies delta once then holds).
    Episode length still counts macro-steps; shaped reward computed over the macro-step.
  - `env.reset_settle_steps` (default 0): hold position for N steps after reset (before sampling
    the EE-relative target) so the arm reaches gravity equilibrium and the target is anchored to a
    stable start pose.
- **Fix (config):** `control_substeps: 8`, `reset_settle_steps: 10`.
- **Fix (tool `probe_actions.py`):** added `--settle` phase + a noop-baseline-subtracted "Net"
  column for honest verdicts.
- **Result:** noop z-drift dropped from ~0.0099 to ~0.0001 (arm now HOLDS), and `+z` finally
  climbs. But x/y/z authority was now wildly anisotropic (y ~63%/step, x ~5-17%, z ~1-2%) —
  revealed Problem 5.

### Problem 5 — Redundant IK drives `wrist_flex` into its limit
- **Tool:** added a "REACHABILITY STRESS TEST" to `probe_actions.py` — hold a large fixed target
  (0.06 m) per axis for many sim steps, report achieved fraction + any joint at a limit.
- **Symptom:** `frac` ~0.22 (+x) / 0.06 (-z), with **`wrist_flex` pinned at its HIGH limit
  (+1.66 ≈ max 1.658) in every test**, even while just holding position.
- **Diagnosis:** 5 IK joints solving a 3-DOF position task = 2 redundant DOFs with nothing
  constraining the null space; the DLS solution drives `wrist_flex` to its stop, which then
  kills x/z authority. (Note: `wrist_flex` DOES affect EE position, but it is *redundant* for
  position; `wrist_roll` barely affects position. 3 DOF — pan+lift+elbow — fully span 3D position.)
- **Fix (config `action.ik_joint_names`):** `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll]` -> `[shoulder_pan, shoulder_lift, elbow_flex]`. Wrists now held at reset angle
  (fixed gripper tilt, irrelevant to a position-only reach).
- **Result:** no joints at limit; reachability `frac` x≈0.84-0.95, y≈0.92, z≈0.45-0.54. Best the
  env has been. Residual: z ~50% with `+z`/x coupling — arm geometry near the vertical workspace
  boundary (manipulability, not saturation). Adding `wrist_flex` back reintroduces the saturation,
  so it is a net loss.

---

# Current config state (`config_ik_sbq.yaml`)

- `env`: `success_threshold 0.03`, `max_episode_steps 80`, `reset_settle_steps 10`,
  `ee_relative_target { enabled, radius 0.08, min_radius 0.04 }`.
- `reward`: shaped; `raw_reward_scale 0.0`, `progress_scale 10.0`, `distance_scale 1.0`,
  `success_bonus 10.0`.
- `action`: `step_size 0.01` + `adaptive_step_size` schedule; `include_noop true`;
  `control_substeps 8`; `ik_command_mode` defaults to `absolute_base` (not set explicitly);
  `ik_joint_names [shoulder_pan, shoulder_lift, elbow_flex]`; `ik_method dls`,
  `ik_params { lambda_val 0.01 }`.
- `observation.position_error`: `edges [-0.20,-0.10,-0.05,-0.025,-0.01,0.01,0.025,0.05,0.10,0.20]`,
  `distance_scale [0.05,0.05,0.05]` (11 bins/axis -> 11^3 states; zero-error bin = index 5).
- `agent` (sbq, euclidean): `gamma 0.99`, `lr 0.1`, epsilon `0.7->0.05` over 120k,
  `beta_0 1.0`, `p 0.5`, `kernel_lambda 0.5`, `search_radius 1.5`, `max_neighbor_delta 3`,
  `alpha_k 0.3`, `use_local_normalization true`.
- `training`: `total_steps 300000`, `eval_interval 10000`, `eval_episodes 20`, `eval_max_steps 80`.
- `logging.run_name: so101_reach_ik_sbq_3dof_seed0`.

# Code changes made this session

- `env_ik.py`:
  - `ik_command_mode` (`absolute_base` default) + absolute base-frame command in `step()` +
    `_ee_pos_b()` helper.
  - `control_substeps` loop in `step()`.
  - `reset_settle_steps` + `_settle()` called in `reset()` (before target resampling).
- `agents.py` (`EuclideanSBQAgent`): `center_bins` + sign-aware `get_neighborhood`.
- `runner.py`: re-enabled greedy eval block in `train()`.
- `config_ik_sbq.yaml`: annealing params, eval on, `reset_settle_steps 10`, `control_substeps 8`,
  3-DOF `ik_joint_names`, fresh run names.
- `probe_actions.py` (NEW): per-action direction probe with settle + noop-baseline Net columns,
  plus a reachability stress test with joint-limit reporting.

# How to run

Probe (verify env before training):
```
/workspace/isaaclab/isaaclab.sh -p -m applications.so101_reach.probe_actions \
  --config applications/so101_reach/config_ik_sbq.yaml --headless
```
Train:
```
/workspace/isaaclab/isaaclab.sh -p -m applications.so101_reach.train_isaac \
  --config applications/so101_reach/config_ik_sbq.yaml --headless
```
Outputs in `outputs/<run_name>/` (`metrics.csv` training rows, `eval_metrics.csv` greedy eval).
Note: Docker runs as root, so output dirs are root-owned.

---

# Next steps / open items

- **Retrain** under `so101_reach_ik_sbq_3dof_seed0` and check `eval_metrics.csv`:
  `eval_distance_mean` should drop below ~0.08 and keep falling; `eval_success_rate` should
  climb above the old ~0.05-0.10 floor; training `|td_error|` should shrink from ~0.72.
- If it plateaus, confirm whether residual failures are z-heavy (straight-up) targets. If so, the
  remaining lever is the **reset posture** (start the arm lower/more crouched so "up" targets sit
  in higher-manipulability space) — settable via env cfg init joint angles (in-scope; no robot
  asset edits). Inspect current reset joint angles first.
- Fair baseline: `config_ik_qlearning.yaml` still has the OLD 5-joint IK set, `control_substeps 1`,
  no `reset_settle_steps`. Sync the three env fixes (3-DOF joints, control_substeps, settle) before
  comparing Q-learning vs SBQ.

# Earlier history (pre-session, kept for reference)

- `so101_reach_ik_sbq_seed0` (1M then 500k steps): ~13-15% success but much was reset-proximal
  luck; no stable trend. Run dir later contaminated (header-only overwrite, stale checkpoints).
- `so101_reach_ik_sbq_shell_seed0` (300k): shell sampling `[0.04, 0.08]` removed trivial starts;
  success real but low/flat ~7-8%, mean failure distance ~0.11 m. This is what motivated the
  session above.
- Artifact hygiene: use a fresh `run_name` per experiment; avoid `--checkpoint latest` when old
  checkpoints remain in a run dir.
```
