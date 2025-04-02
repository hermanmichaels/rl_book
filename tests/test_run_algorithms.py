from typing import Callable

import pytest

from rl_book.dp import policy_iteration, value_iteration
from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from rl_book.planning import dyna_q, mcts, prioritized_sweeping
from rl_book.td import double_q, q, sarsa
from rl_book.td_n import sarsa_n, tree_n


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
