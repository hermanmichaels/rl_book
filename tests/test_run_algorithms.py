from typing import Callable

import pytest

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from gym_utils import get_observation_action_space
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from planning import dyna_q, mcts, prioritized_sweeping
from td import double_q, q, sarsa
from td_n import sarsa_n, tree_n


@pytest.mark.parametrize("method", [policy_iteration, value_iteration])
def test_dp(env_train: ParametrizedEnv, method: Callable):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = method(env_train)
    assert pi.shape == (observation_space.n,)


@pytest.mark.parametrize(
    "method", [mc_es, on_policy_mc, off_policy_mc, off_policy_mc_non_inc]
)
def test_mc(env_train: ParametrizedEnv, method: Callable):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = method(env_train)
    assert pi.shape == (observation_space.n,)


@pytest.mark.parametrize("method", [sarsa, q, double_q])
def test_td(env_train: ParametrizedEnv, method: Callable):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = method(env_train)
    assert pi.shape == (observation_space.n,)


@pytest.mark.parametrize("method", [sarsa_n, tree_n])
def test_td_n(env_train: ParametrizedEnv, method: Callable):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = method(env_train)
    assert pi.shape == (observation_space.n,)


@pytest.mark.parametrize(
    "method",
    [
        dyna_q,
        prioritized_sweeping,
    ],
)
def test_planning(env_train: ParametrizedEnv, method: Callable):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = method(env_train)
    assert pi.shape == (observation_space.n,)


def test_mcts(env_train: ParametrizedEnv):
    action = mcts(env_train, [])
    assert isinstance(action, int)
