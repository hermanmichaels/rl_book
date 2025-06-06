import argparse

import gymnasium as gym

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.mc import OffPolicyMC, OffPolicyMCNonInc, OnPolicyMC, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa, double_q, expected_sarsa, q, sarsa
from rl_book.methods.td_n import SarsaN, TreeN, sarsa_n, tree_n
from rl_book.replay_utils import ReplayItem
from rl_book.utils import get_policy

GAMMA = 0.97
EPS = 0.001
NUM_STEPS = 100


def train(env, method):
    observation_space, action_space = get_observation_action_space(env)

    for step in range(10000):
        # print(step)
        observation, _ = env.env.reset()
        terminated = truncated = False

        cur_episode_len = 0
        episode = []

        while not terminated and not truncated:
            action = method.act(observation, [1 for _ in range(action_space.n)], step)

            # import ipdb
            # ipdb.set_trace()
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            episode.append(ReplayItem(observation, action, reward, [1 for _ in range(action_space.n)]))
            method.update(episode)

            observation = observation_new

            cur_episode_len += 1
            # print(cur_episode_len)

        episode.append(ReplayItem(observation_new, -1, reward, []))
        method.finalize(episode, step)

    pi = get_policy(method.Q, observation_space, action_space) # todo: make function of class

    return False, pi, step


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
    env_train = ParametrizedEnv(
        gym_env_train, GAMMA, intermediate_rewards=True, eps_decay=True
    )

    # Find policy
    if method_name == "policy_iteration":
        pi = policy_iteration(env_train)[1]
    elif method_name == "value_iteration":
        pi = value_iteration(env_train)[1]
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
        else:
            raise ValueError(f"Unknown solution method {method_name}")
        
        pi = train(env_train, method)[1]

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
