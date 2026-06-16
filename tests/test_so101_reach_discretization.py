from __future__ import annotations

import numpy as np

from applications.so101_reach.action_space import CartesianStepActionMapper, JointStepActionMapper
from applications.so101_reach.agents import EuclideanSBQAgent
from applications.so101_reach.discretization import StateDiscretizer
from core.spaces import DiscreteActionSpace


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def test_state_discretizer_uses_dense_center_bins():
    discretizer = StateDiscretizer.from_config({}, JOINTS)
    near_zero = discretizer.encode([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    small_positive = discretizer.encode([0.015, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    large_positive = discretizer.encode([0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])

    assert discretizer.observation_space.nvec[:3] == [11, 11, 11]
    assert near_zero[0] == 5
    assert small_positive[0] > near_zero[0]
    assert large_positive[0] == 10


def test_joint_step_action_mapper_has_noop_and_signed_joint_steps():
    mapper = JointStepActionMapper(JOINTS, step_size=0.05)

    assert mapper.action_space.n == 11
    assert mapper.action_names[0] == "noop"
    np.testing.assert_allclose(mapper.to_continuous(0), np.zeros(5))
    np.testing.assert_allclose(mapper.to_continuous(1), [0.05, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(mapper.to_continuous(2), [-0.05, 0.0, 0.0, 0.0, 0.0])


def test_xyz_only_discretizer_has_small_task_space_state():
    discretizer = StateDiscretizer.from_config({}, [])

    assert discretizer.observation_space.nvec == [11, 11, 11]
    assert discretizer.observation_space.size == 1331
    assert discretizer.feature_names == ["target_error_x", "target_error_y", "target_error_z"]


def test_cartesian_step_action_mapper_has_noop_and_signed_xyz_steps():
    mapper = CartesianStepActionMapper(step_size=0.02)

    assert mapper.action_space.n == 7
    assert mapper.action_names == ["noop", "+x", "-x", "+y", "-y", "+z", "-z"]
    np.testing.assert_allclose(mapper.to_continuous(0), np.zeros(3))
    np.testing.assert_allclose(mapper.to_continuous(1), [0.02, 0.0, 0.0])
    np.testing.assert_allclose(mapper.to_continuous(6), [0.0, 0.0, -0.02])


def test_euclidean_sbq_neighbors_are_distance_limited():
    discretizer = StateDiscretizer.from_config({}, JOINTS)
    agent = EuclideanSBQAgent(
        observation_space=discretizer.observation_space,
        action_space=DiscreteActionSpace(3),
        rng=np.random.default_rng(0),
        distance_centers=discretizer.distance_centers,
        search_radius=0.5,
        max_neighbor_delta=1,
    )
    center_state = discretizer.observation_space.to_index(discretizer.encode([0.0, 0.0, 0.0], [0, 0, 0, 0, 0]))
    neighbors = agent.get_neighborhood(center_state)

    assert center_state in set(neighbors.tolist())
    assert len(neighbors) > 1
    assert all(agent.get_distance(center_state, int(idx)) <= 0.5 for idx in neighbors)

