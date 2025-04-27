import time
from functools import partial

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.core import Env

from rl_book.env import ParametrizedEnv
from rl_book.methods.mc import mc_es, off_policy_mc, on_policy_mc
from rl_book.methods.planning import dyna_q, prioritized_sweeping
from rl_book.methods.td import double_q, expected_sarsa, q, sarsa
from rl_book.methods.td_n import sarsa_n, tree_n

GAMMA = 0.97


MAX_INFERENCE_STEPS = 1000

MAX_STEPS = [10000, 30000, 100000]
# MAX_STEPS = [100, 1000]
TRIES_PER_STEP = 3


def generate_random_env(n: int, extra_rewards, eps_decay) -> ParametrizedEnv:
    desc = generate_random_map(size=n)
    gym_env = gym.make(
        "FrozenLake-v1", desc=desc, is_slippery=False, render_mode="human"
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


def plot_results(needed_steps, methods, min_grid_size, max_grid_size, fig_path):
    x_values = [n for n in range(min_grid_size, max_grid_size)]
    markers = ["o", "s", "^", "*"]

    labels = ["baseline", "int. r", "eps decay", "int. r + eps decay"]
    for idx, y_values in enumerate(needed_steps):
        plt.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            label=methods[idx].__name__,
        )
        plt.legend()
        plt.xticks([5, 10, 15])
        plt.xlabel("Gridworld size")
        plt.ylabel("Steps needed")
        plt.savefig(fig_path)
    plt.clf()


def benchmark(
    methods,
    min_grid_size=4,
    max_grid_size=10,
    extra_rewards: bool = False,
    eps_decay: bool = False,
    fig_path: str = "result.png",
) -> None:
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
                # TODO: name wrong
                steps_needed_cur = []
                for _ in range(3):
                    # TODO: stop if previous run had less steps
                    env = generate_random_env(n, extra_rewards, eps_decay)
                    # TODO: env param order wrong
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
            # TODO: add dummy if no success

            print(f"{method} -- {steps_needed_cur}")

        # TODO: format
        print(f"Finished benchmarking grid size {n} x {n} in {time.time() - start}s")

    plot_results(steps_needed, methods, min_grid_size, max_grid_size, fig_path)


if __name__ == "__main__":
    # benchmark([policy_iteration, value_iteration], fig_path="results/dp.png")
    # benchmark(
    #    [mc_es, on_policy_mc, off_policy_mc], 5, 16, False, False,
    #    fig_path="results/mc.png",
    # )
    # benchmark(
    #    [mc_es, on_policy_mc, off_policy_mc], 5, 16, True, True,
    #    fig_path="results/mc-true.png",
    # )
    # benchmark([q, q, q, q], 9, 17, False, fig_path="results/q.png")
    # benchmark([sarsa, q, expected_sarsa, double_q], 5, 16, False, False, fig_path="results/td.png")
    # benchmark([sarsa, q, expected_sarsa, double_q], 5, 16, True, True, fig_path="results/td-true.png")
    # benchmark([sarsa, q, double_q], 5, 26, True, True, fig_path="results/td.png")
    # benchmark([sarsa_n, tree_n], 5, 16, False, False, fig_path="results/td_n_.png")
    # benchmark([sarsa_n, tree_n], 5, 16, True, True, fig_path="results/td_n_true.png")
    benchmark(
        [dyna_q, prioritized_sweeping],
        50,
        51,
        False,
        False,
        fig_path="results/planning.png",
    )
