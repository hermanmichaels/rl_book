from typing import Callable

import pytest

from rl_book.env import ConnectFourEnv, GridWorldEnv, ParametrizedEnv, TicTacToeEnv
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.mc import OffPolicyMC, OnPolicyMC
from rl_book.methods.method import MethodWithStats
from rl_book.methods.misc import Random
from rl_book.methods.planning import DynaQ, mcts
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rl_book.methods.td_n import SarsaN, TreeN
from rl_book.methods.training import train_multi_player, train_single_player

MAX_STEPS = 100


@pytest.mark.parametrize("method", [policy_iteration, value_iteration])
def test_dp_grid_world(grid_world_env: GridWorldEnv, method: Callable):
    method(grid_world_env)


@pytest.mark.parametrize(
    "method_name",
    [
        OnPolicyMC,
        OffPolicyMC,
        Sarsa,
        QLearning,
        ExpectedSarsa,
        DoubleQ,
        SarsaN,
        TreeN,
        DynaQ,
    ],
)
def test_methods_grid_world(grid_world_env: GridWorldEnv, method_name: Callable):
    method = method_name(grid_world_env)
    train_single_player(grid_world_env, method)[1]


@pytest.mark.parametrize("method_name", [OnPolicyMC, OffPolicyMC, Sarsa, QLearning, ExpectedSarsa, SarsaN, TreeN])
def test_methods_tic_tac_toe(tic_tac_toe_env: TicTacToeEnv, method_name: Callable):
    zoo = [MethodWithStats(Random(tic_tac_toe_env))]
    methods = [MethodWithStats(method_name(tic_tac_toe_env))]
    train_multi_player(tic_tac_toe_env, methods, zoo, max_steps=100)

@pytest.mark.parametrize("method_name", [OnPolicyMC, OffPolicyMC, Sarsa, QLearning, ExpectedSarsa, SarsaN, TreeN])
def test_methods_connect_four(connect_four_env: ConnectFourEnv, method_name: Callable):
    zoo = [MethodWithStats(Random(connect_four_env))]
    methods = [MethodWithStats(method_name(connect_four_env))]
    train_multi_player(connect_four_env, methods, zoo, max_steps=100)


def test_mcts(grid_world_env: GridWorldEnv):
    action = mcts(grid_world_env, [])
    assert isinstance(action, int)
