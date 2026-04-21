import gymnasium as gym
import pytest
import torch
from pettingzoo.classic import connect_four_v3, tictactoe_v3

from rl_book.env import ConnectFourEnv, GridWorldEnv, ObsMode, TicTacToeEnv


@pytest.fixture
def grid_world_env() -> GridWorldEnv:
    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    return GridWorldEnv(
        gym_env_train, 0.99, False, False, ObsMode.DEFAULT, torch.device("cpu")
    )


@pytest.fixture
def tic_tac_toe_env() -> TicTacToeEnv:
    return TicTacToeEnv(tictactoe_v3.env(), 0.99)


@pytest.fixture
def connect_four_env() -> ConnectFourEnv:
    return ConnectFourEnv(connect_four_v3.env(), 0.99)
