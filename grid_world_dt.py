import argparse

import gymnasium as gym

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from planning import dyna_q, mcts, prioritized_sweeping
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

    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    desc = generate_random_map(size=5)

    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=desc,
        # map_name="4x4",
        is_slippery=False,
    )
    env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)

    # Find policy
    if method == "policy_iteration":
        pi = q(env_train)

    gym_env_train.close()

    gym_env_test = gym.make(
        "FrozenLake-v1",
        desc=desc,
        # map_name="4x4",
        is_slippery=False,
        render_mode="human",
    )

    gym_env_train = gym.make(
        "FrozenLake-v1",
        # desc=None,
        desc=desc,
        # map_name="4x4",
        is_slippery=False,
    )
    env_train = ParametrizedEnv(gym_env_train, 0.9, EPS)

    # Test policy and visualize found solution
    observation, _ = gym_env_test.reset()

    print(observation)

    actions = []

    for _ in range(NUM_STEPS):
        action = mcts(env_train, pi, actions)
        actions.append(action)
        # import random
        # action = random.randint(0, 3)
        observation, _, terminated, truncated, _ = gym_env_test.step(action)
        print(f"{action} -> {observation}")
        # import ipdb
        # ipdb.set_trace()
        if terminated or truncated:
            break
    gym_env_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument("--method", type=str, required=True, help="A string input")
    args = parser.parse_args()

    solve_grid_world(args.method)
