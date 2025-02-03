import argparse

from grid_world_utils import find_policy
import gymnasium as gym
from gymnasium.core import Env

from env import ParametrizedEnv
from planning import mcts

GAMMA = 0.9
EPS = 0.001
NUM_STEPS = 100


def solve_grid_world(method: str) -> None:
    """Solve the grid world problem using the chosen solving method.

    Args:
        method: solving method
    """
    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)

    # Find policy
    pi = find_policy(env_train, method)

    gym_env_train.close()

    gym_env_test = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="human",
    )

    # Test policy and visualize found solution
    observation, _ = gym_env_test.reset()
    actions = []
    for _ in range(NUM_STEPS):
        action = pi[observation]
        action = mcts(env_train, pi, actions)
        actions.append(action)
        observation, _, terminated, truncated, _ = gym_env_test.step(action)
        if terminated or truncated:
            break
    gym_env_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--method", type=str, required=True, help="A string input")
    args = parser.parse_args()

    solve_grid_world(args.method)
