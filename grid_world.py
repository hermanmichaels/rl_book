import argparse

import gymnasium as gym

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv

GAMMA = 0.97
EPS = 0.001
NUM_STEPS = 100


def solve_grid_world(method: str) -> None:
    """Solve the grid world problem using the chosen solving method.

    Args:
        method: solving method
    """
    gym_env = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="human",
    )
    env = ParametrizedEnv(gym_env, GAMMA, EPS)

    # Find policy
    if method == "policy_iteration":
        pi = policy_iteration(env)
    elif method == "value_iteration":
        pi = value_iteration(env)
    else:
        raise ValueError(f"Unknown solution method {method}")

    # Test policy and visualize found solution
    observation, _ = env.env.reset()
    for _ in range(NUM_STEPS):
        action = pi[observation]
        observation, _, terminated, truncated, _ = env.env.step(action)
        if terminated or truncated:
            break
    env.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--method", type=str, required=True, help="A string input")
    args = parser.parse_args()
    
    solve_grid_world(args.method)
