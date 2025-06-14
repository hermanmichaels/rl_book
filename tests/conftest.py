import gymnasium as gym
import pytest

from rl_book.env import GridWorldEnv, TicTacToeEnv
from pettingzoo.classic import connect_four_v3, tictactoe_v3

@pytest.fixture
def grid_world_env() -> GridWorldEnv:
    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    return GridWorldEnv(gym_env_train, 0.99, False, False)

@pytest.fixture
def tic_tac_toe_env() -> TicTacToeEnv:
    return TicTacToeEnv(tictactoe_v3.env(), 0.99)
