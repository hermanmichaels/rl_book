from typing import Callable

import pytest

from rl_book.env import GridWorldEnv, ParametrizedEnv, TicTacToeEnv
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.mc import OffPolicyMC, OffPolicyMCNonInc, OnPolicyMC
from rl_book.methods.method import MethodWithStats
from rl_book.methods.misc import Random
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rl_book.methods.td_n import SarsaN, TreeN
from rl_book.methods.training import train_multi_player, train_single_player

MAX_STEPS = 100


@pytest.mark.parametrize("method", [policy_iteration, value_iteration])
def test_dp_grid_world(grid_world_env: GridWorldEnv, method: Callable):
    num_states = grid_world_env.get_observation_space_len()
    pi = method(grid_world_env, max_steps=MAX_STEPS)
    assert pi.shape == (num_states,)


@pytest.mark.parametrize("method_name", [OnPolicyMC, OffPolicyMC, OffPolicyMCNonInc, Sarsa, QLearning, ExpectedSarsa, SarsaN, TreeN])
def test_methods_grid_world(grid_world_env: GridWorldEnv, method_name: Callable):
    num_states = grid_world_env.get_observation_space_len()
    method = method_name(grid_world_env)
    pi = train_single_player(grid_world_env, method)[1]
    assert pi.shape == (num_states)

# , OffPolicyMC, OffPolicyMCNonInc, Sarsa, QLearning, ExpectedSarsa, SarsaN, TreeN
@pytest.mark.parametrize("method_name", [OnPolicyMC])
def test_methods_tic_tac_toe(tic_tac_toe_env: TicTacToeEnv, method_name: Callable):
    num_states = tic_tac_toe_env.get_observation_space_len()
    zoo = [MethodWithStats(Random(tic_tac_toe_env))]
    methods= [MethodWithStats(method_name(tic_tac_toe_env))]
    train_multi_player(tic_tac_toe_env, methods, zoo, max_steps=100)
    # assert pi.shape == (num_states)


# @pytest.mark.parametrize(
#     "method",
#     [
#         dyna_q,
#         prioritized_sweeping,
#     ],
# )
# def test_planning(env_train: ParametrizedEnv, method: Callable):
#     observation_space, _ = get_observation_action_space(env_train)
#     pi = method(env_train, max_steps=MAX_STEPS)[1]
#     assert pi.shape == (observation_space.n,)


# def test_mcts(env_train: ParametrizedEnv):
#     action = mcts(env_train, [])
#     assert isinstance(action, int)
