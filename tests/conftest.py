from env import ParametrizedEnv
import pytest
import gymnasium as gym

@pytest.fixture
def env_train() -> ParametrizedEnv:
    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    return ParametrizedEnv(gym_env_train, 0.99, 0.05)