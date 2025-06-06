import time
from functools import partial
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rl_book.env import ParametrizedEnv
from rl_book.methods.dp import policy_iteration, value_iteration
from rl_book.methods.mc import mc_es, off_policy_mc, on_policy_mc
from rl_book.methods.planning import dyna_q, prioritized_sweeping
from rl_book.methods.td import double_q, expected_sarsa, q, sarsa
from rl_book.methods.td_n import sarsa_n, tree_n

GAMMA = 0.97
MAX_INFERENCE_STEPS = 1000
MAX_STEPS = [10000, 30000, 100000, 200000]
TRIES_PER_STEP = 3


def generate_random_env(
    n: int, extra_rewards: bool, eps_decay: bool
) -> ParametrizedEnv:
    desc = generate_random_map(size=n)
    gym_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
    )
    return ParametrizedEnv(
        gym_env, GAMMA, intermediate_rewards=extra_rewards, eps_decay=eps_decay
    )


def get_check_frequency(step: int) -> int:
    if step < 1000:
        return 100
    elif step < 10000:
        return 1000
    else:
        return 10000


def success_callback(pi: np.ndarray, step: int, env: Env) -> bool:
    """Tests whether the given policy can successfully solve the given Gridworld
    environment.

    Args:
        pi: policy
        step (int): current step
        env: env

    Returns:
        False if current step is not a step to be checked, or policy does not solve env
        - True otherwise.
    """
    if step % get_check_frequency(step) != 0:
        return False

    observation, _ = env.reset()
    for _ in range(MAX_INFERENCE_STEPS):
        action = pi[observation]
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    return reward == 1


def plot_results(
    needed_steps: list[list[int]],
    methods: list[Callable],
    min_grid_size: int,
    max_grid_size: int,
    fig_path: str,
) -> None:
    x_values = [n for n in range(min_grid_size, max_grid_size)]
    markers = ["o", "s", "^", "*"]

    for idx, y_values in enumerate(needed_steps):
        plt.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            label=methods[idx].__name__,
        )
        plt.legend()
        plt.xlabel("Gridworld size")
        plt.ylabel("Steps needed")
        plt.savefig(fig_path)
    plt.clf()


def benchmark(
    methods: list[Callable],
    min_grid_size=3,
    max_grid_size=8,
    extra_rewards: bool = True,
    eps_decay: bool = True,
    fig_path: str = "result.png",
) -> None:
    """Runs a benchmarking job.

    Args:
        methods: methods to run
        min_grid_size: starting Gridworld size
        max_grid_size: ending Gridworld size
        extra_rewards: use intermediate rewards
        eps_decay: decay epsilon
        fig_path: path to which to save the figure to
    """
    steps_needed: list[list[int]] = [[] for _ in range(len(methods))]

    # Iterate over all possible grid sizes.
    for n in range(min_grid_size, max_grid_size):
        start = time.time()
        # Iterate over all methods.
        for idx, method in enumerate(methods):
            # For faster results and reduced variance (e.g. unlucky initialization)
            # try increasing maximal number of steps, and run multiple trainings
            # with each threshold - then store the best run.
            found_sol = False
            for max_steps in MAX_STEPS:
                steps_needed_cur: list[int] = []
                for _ in range(TRIES_PER_STEP):
                    env = generate_random_env(n, extra_rewards, eps_decay)
                    callback = partial(success_callback, env=env.env)
                    max_s = (
                        max_steps + 1
                        if not steps_needed_cur
                        else max(1, min(steps_needed_cur))
                    )
                    success, _, step = method(env, callback, max_s)
                    if success:
                        steps_needed_cur.append(step)
                if steps_needed_cur:
                    steps_needed[idx].append(min(steps_needed_cur))
                    found_sol = True
                    break

            if not found_sol:
                steps_needed[idx].append(MAX_STEPS[-1])

            print(f"{method}, steps needed: {steps_needed_cur}")

        print(f"Finished benchmarking grid size {n} x {n} in {time.time() - start}s")

    plot_results(steps_needed, methods, min_grid_size, max_grid_size, fig_path)


if __name__ == "__main__":
    benchmark([policy_iteration, value_iteration], fig_path="results/dp.png")
    benchmark(
        [mc_es, on_policy_mc, off_policy_mc],
        fig_path="results/mc.png",
    )
    benchmark([sarsa, q, expected_sarsa, double_q], fig_path="results/td.png")
    benchmark([sarsa_n, tree_n], fig_path="results/td_n_.png")
    benchmark(
        [dyna_q, prioritized_sweeping],
        fig_path="results/planning.png",
    )
