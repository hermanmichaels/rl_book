import time
from functools import partial

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from planning import dyna_q, prioritized_sweeping
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n

GAMMA = 0.97
EPS = 0.001


MAX_INFERENCE_STEPS = 1000

MAX_STEPS = [10001, 30001, 100001, 1000001]
TRIES_PER_STEP = 3


def generate_random_env(n: int) -> ParametrizedEnv:
    desc = generate_random_map(size=n)
    gym_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
    )
    return ParametrizedEnv(gym_env, GAMMA, EPS)


def get_check_frequency(step: int) -> int:
    if step < 1000:
        return 100
    elif step < 10000:
        return 1000
    else:
        return 10000


def success_callback(pi: np.ndarray, step: int, env: ParametrizedEnv) -> bool:
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

    for idx, y_values in enumerate(needed_steps):
        plt.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            label=methods[idx].__name__,
        )
        plt.legend()
        plt.savefig(fig_path)
    plt.clf()


def benchmark(
    methods, min_grid_size=4, max_grid_size=14, fig_path: str = "result.png"
) -> None:
    needed_steps = [[] for _ in range(len(methods))]

    # Iterate over all possible grid sizes.
    for n in range(min_grid_size, max_grid_size):
        start = time.time()
        # Iterate over all methods.
        for idx, method in enumerate(methods):
            # For faster results and reduced variance (e.g. unlucky initialization)
            # try increasing maximal number of steps, and run multiple trainings
            # with each threshold - then store the best run.
            # TODO: plot variance / confidence interval?
            for max_steps in MAX_STEPS:
                # TODO: name wrong
                needed_steps_cur_method = []
                for _ in range(TRIES_PER_STEP):
                    env = generate_random_env(n)
                    # TODO: env param order wrong
                    callback = partial(success_callback, env=env.env)
                    success, _, step = method(env, callback, max_steps)
                    if not success:
                        step = 10000000
                    if True or success:
                        needed_steps_cur_method.append(step)
                        # break # TODO: remove for lesser steps
                print(needed_steps_cur_method)
                if needed_steps_cur_method and min(needed_steps_cur_method) != 10000000:
                    needed_steps[idx].append(min(needed_steps_cur_method))
                    break
        # TODO: format
        print(f"Finished benchmarking grid size {n} x {n} in {time.time() - start}s")

    plot_results(needed_steps, methods, min_grid_size, max_grid_size, fig_path)


if __name__ == "__main__":
    # benchmark([policy_iteration, value_iteration], fig_path="res_dp.png")
    # benchmark(
    #    [mc_es, on_policy_mc, off_policy_mc, off_policy_mc_non_inc],
    #    fig_path="res_mc.png",
    # )
    benchmark([q], fig_path="res_td.png")
    # benchmark([sarsa, q, expected_sarsa, double_q], fig_path="res_td.png")
    # benchmark([sarsa_n, tree_n], fig_path="res_td_n.png")
    # benchmark([dyna_q, prioritized_sweeping], fig_path="res_planning.png")
