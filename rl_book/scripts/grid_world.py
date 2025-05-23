import argparse

import gymnasium as gym

from rl_book.env import ParametrizedEnv
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from rl_book.methods.td import double_q, expected_sarsa, q, sarsa
from rl_book.methods.td_n import sarsa_n, tree_n

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
    env_train = ParametrizedEnv(
        gym_env_train, GAMMA, intermediate_rewards=True, eps_decay=True
    )

    # Find policy
    if method == "policy_iteration":
        pi = policy_iteration(env_train)[1]
    elif method == "value_iteration":
        pi = value_iteration(env_train)[1]
    elif method == "mc_es":
        pi = mc_es(env_train)[1]
    elif method == "on_policy_mc":
        pi = on_policy_mc(env_train)[1]
    elif method == "off_policy_mc":
        pi = off_policy_mc(env_train)[1]
    elif method == "off_policy_mc_non_inc":
        pi = off_policy_mc_non_inc(env_train)[1]
    elif method == "sarsa":
        pi = sarsa(env_train)[1]
    elif method == "q":
        pi = q(env_train)[1]
    elif method == "expected_sarsa":
        pi = expected_sarsa(env_train)[1]
    elif method == "double_q":
        pi = double_q(env_train)[1]
    elif method == "sarsa_n":
        pi = sarsa_n(env_train)[1]
    elif method == "tree_n":
        pi = tree_n(env_train)[1]
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
    parser = argparse.ArgumentParser(description="Solve Gridworld with RL")
    parser.add_argument("--method", type=str, required=True, help="Solution method")
    args = parser.parse_args()

    solve_grid_world(args.method)
