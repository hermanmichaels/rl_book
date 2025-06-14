import argparse

import gymnasium as gym

from rl_book.env import GridWorldEnv
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.inference import test_single_player
from rl_book.methods.mc import OffPolicyMC, OffPolicyMCNonInc, OnPolicyMC
# from rl_book.methods.planning import DynaQ
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rl_book.methods.td_n import SarsaN, TreeN
from rl_book.methods.training import train_single_player

GAMMA = 0.97
EPS = 0.001
NUM_STEPS = 100


def solve_grid_world(method_name: str) -> None:
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
    env_train = GridWorldEnv(
        gym_env_train, GAMMA, intermediate_rewards=True, eps_decay=True
    )

    # Find policy
    if method_name == "policy_iteration":
        method = policy_iteration(env_train, 100)
    elif method_name == "value_iteration":
        method = value_iteration(env_train, 100)
    else:
        if method_name == "on_policy_mc":
            method = OnPolicyMC(env_train)
        elif method_name == "off_policy_mc":
            method = OffPolicyMC(env_train)
        elif method_name == "off_policy_mc_non_inc":
            method = OffPolicyMCNonInc(env_train)
        elif method_name == "sarsa":
            method = Sarsa(env_train)
        elif method_name == "q":
            method = QLearning(env_train)
        elif method_name == "expected_sarsa":
            method = ExpectedSarsa(env_train)
        elif method_name == "double_q":
            method = DoubleQ(env_train)
        elif method_name == "sarsa_n":
            method = SarsaN(env_train)
        elif method_name == "tree_n":
            method = TreeN(env_train)
        elif method_name == "dyna_q":
            method = DynaQ(env_train)
        else:
            raise ValueError(f"Unknown solution method {method_name}")

        train_single_player(env_train, method)[1]

    gym_env_test = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="human",
    )

    # Test policy and visualize found solution
    test_single_player(gym_env_test, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Gridworld with RL")
    parser.add_argument("--method", type=str, required=True, help="Solution method")
    args = parser.parse_args()

    solve_grid_world(args.method)
