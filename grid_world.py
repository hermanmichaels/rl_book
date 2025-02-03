import argparse

import gymnasium as gym

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n

GAMMA = 0.97
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
    if method == "policy_iteration":
        pi = policy_iteration(env_train)
    elif method == "value_iteration":
        pi = value_iteration(env_train)
    elif method == "mc_es":
        pi = mc_es(env_train)
    elif method == "on_policy_mc":
        pi = on_policy_mc(env_train)
    elif method == "off_policy_mc":
        pi = off_policy_mc(env_train)
    elif method == "off_policy_mc_non_inc":
        pi = off_policy_mc_non_inc(env_train)
    elif method == "sarsa":
        pi = sarsa(env_train)
    elif method == "q":
        pi = q(env_train)
    elif method == "expected_sarsa":
        pi = expected_sarsa(env_train)
    elif method == "double_q":
        pi = double_q(env_train)
    elif method == "sarsa_n":
        pi = sarsa_n(env_train)
    elif method == "tree_n":
        pi = tree_n(env_train)
    else:
        raise ValueError(f"Unknown solution method {method}")
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
    for _ in range(NUM_STEPS):
        action = pi[observation]
        observation, _, terminated, truncated, _ = gym_env_test.step(action)
        if terminated or truncated:
            break
    gym_env_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--method", type=str, required=True, help="A string input")
    args = parser.parse_args()

    solve_grid_world(args.method)
