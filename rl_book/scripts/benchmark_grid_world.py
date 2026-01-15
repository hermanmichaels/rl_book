import time
from functools import partial
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rl_book.env import GridWorldEnv, ObsMode
from rl_book.methods.mc import OffPolicyMC, OnPolicyMC
from rl_book.methods.method import RLMethod
from rl_book.methods.planning import DynaQ
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rl_book.methods.td_approx import (SemiGradientSarsaCNN,
                                       SemiGradientSarsaLinear,
                                       SemiGradientSarsaNCNN)
from rl_book.methods.td_n import SarsaN, TreeN
from rl_book.methods.training import train_single_player

GAMMA = 0.97
MAX_INFERENCE_STEPS = 1000
MAX_STEPS = [10000, 30000, 100000, 200000]
TRIES_PER_STEP = 3


def generate_random_env(
    n: int, extra_rewards: bool, eps_decay: bool, obs_mode: ObsMode, device=torch.device
) -> GridWorldEnv:
    desc = generate_random_map(size=n)
    gym_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
    )
    return GridWorldEnv(
        gym_env,
        GAMMA,
        intermediate_rewards=extra_rewards,
        eps_decay=eps_decay,
        obs_mode=obs_mode,
        device=device,
    )


def get_check_frequency(step: int) -> int:
    if step < 1000:
        return 100
    elif step < 10000:
        return 1000
    else:
        return 10000


def success_callback(method: RLMethod, step: int, env: Env) -> bool:
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

    method.eval()

    observation, _ = env.reset()
    for _ in range(MAX_INFERENCE_STEPS):
        action = method.act(observation, step)
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    method.train()

    return reward == 1


def plot_results(
    needed_steps: list[list[int]],
    methods: list[Callable],
    min_grid_size: int,
    max_grid_size: int,
    fig_path: str,
) -> None:
    x_values = [10, 20, 30, 40, 50]  # [n for n in range(min_grid_size, max_grid_size)]
    markers = ["o", "s", "^", "*"]

    for idx, y_values in enumerate(needed_steps):
        plt.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            label=methods[idx].__name__,
        )
        plt.legend()
        plt.xticks([10, 20, 30, 40, 50])
        plt.xlabel("Gridworld size")
        plt.ylabel("Steps needed")

    plt.savefig(fig_path)
    plt.clf()


def benchmark(
    methods: list,
    min_grid_size=5,
    max_grid_size=26,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rasterized_models = ["SemiGradientSarsaCNN", "SemiGradientSarsaNCNN"]

    # Iterate over all possible grid sizes.
    for n in [50]:
        start = time.time()
        # Iterate over all methods.
        for idx, method_ in enumerate(methods):
            obs_mode = (
                ObsMode.RASTERIZED
                if method_.__name__ in rasterized_models
                else ObsMode.DEFAULT
            )
            # For faster results and reduced variance (e.g. unlucky initialization)
            # try increasing maximal number of steps, and run multiple trainings
            # with each threshold - then store the best run.
            found_sol = False
            for max_steps in MAX_STEPS:
                steps_needed_cur: list[int] = []
                for _ in range(TRIES_PER_STEP):
                    env = generate_random_env(
                        n, extra_rewards, eps_decay, obs_mode, device
                    )
                    if method_.__name__ in rasterized_models:
                        method = method_(env, device=device)
                    else:
                        method = method_(env)
                    callback = partial(success_callback, env=env.env)
                    max_s = (
                        max_steps + 1
                        if not steps_needed_cur
                        else max(1, min(steps_needed_cur))
                    )
                    success, step = train_single_player(env, method, max_s, callback)
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
    # benchmark(
    #     [OnPolicyMC, OffPolicyMC],
    #     fig_path="results/mc.png",
    # )
    # benchmark([Sarsa, QLearning, ExpectedSarsa, DoubleQ], fig_path="results/td.png")
    # benchmark([SarsaN, TreeN], fig_path="results/td_n.png")
    # benchmark(
    #     [DynaQ],
    #     fig_path="results/planning.png",
    # )
    # benchmark(
    #     [SemiGradientSarsaLinear, SemiGradientSarsaCNN, SemiGradientSarsaNCNN],
    #     fig_path="results/td_approx.png",
    # )
    # benchmark(
    #     [OnPolicyMC, Sarsa, QLearning, SarsaN, TreeN, DynaQ],
    #     fig_path="results/td_approx.png",
    # )
    benchmark(
        [SemiGradientSarsaCNN],
        fig_path="results/sarsa_approx.png",
    )
