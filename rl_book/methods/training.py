# done for unit test?
import random
from typing import Callable

import matplotlib.pyplot as plt

from rl_book.env import MultiPlayerEnv, ParametrizedEnv
from rl_book.methods.method import MethodWithStats, RLMethod
from rl_book.pretty_print import log_methods
from rl_book.replay_utils import ReplayItem


def train_single_player(
    env: ParametrizedEnv,
    method: RLMethod,
    max_steps: int = 100,
    callback: Callable | None = None,
) -> tuple[bool, int]:
    """Trains a method on single-player environments.

    Args:
        env: env to use
        method: method to use
        max_steps: maximal number of update steps
        callback: callback to determine if method already solves the given problem

    Returns:
        tuple of success, found policy, number of update steps
    """
    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        episode = []
        cur_episode_len = 0

        while not terminated and not truncated:
            action = method.act(observation, step)

            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            episode.append(ReplayItem(observation, action, reward))
            method.update(episode, step)

            observation = observation_new

            # NOTE: this is highly dependent on environment size
            cur_episode_len += 1
            if cur_episode_len > env.get_max_num_steps():
                break

        method.finalize(episode, step)

        if callback and callback(method, step):
            return True, step

    env.env.close()

    return False, step


def train_multi_player(
    env: MultiPlayerEnv,
    methods: list[MethodWithStats],
    zoo: list[MethodWithStats],
    max_steps: int = 100,
    zoo_update_interval: int = 500,
    zoo_size: int = 50,
    plot_interval: int | None = None,
) -> None:
    """Trains a method on multi-player environments (atm only 2 players are supported).

    Args:
        env: env to use
        methods: methods to train
        zoo: initial list of opponents
        max_steps: maixmal number of update steps
    """
    # For plotting: keep (step, win_ratio) tuples for every method at different steps.
    win_ratios: list[list[tuple[int, float]]] = [[] for _ in methods]

    for step in range(max_steps):
        env.env.reset()

        # Draw random method to update and random start position
        method_idx = random.randint(0, len(methods) - 1)
        player_pos = random.randint(0, 1)

        # Draw random opponent
        opponent_idx = random.randint(0, len(zoo) - 1)
        opponent = zoo[opponent_idx]

        methods[method_idx].update_pick()
        zoo[opponent_idx].update_pick()

        state_dict = {}
        done = False
        episode = []

        while not done:
            agent = env.env.agent_selection  # type: ignore
            (
                observation,
                reward,
                termination,
                truncation,
                _,
            ) = env.env.last()  # type: ignore

            done = termination or truncation

            if done:
                action = None
                # Game over, rewards contains all playerss
                methods[method_idx].update_result(
                    env.get_game_result(
                        env.env.rewards[env.players[player_pos]]  # type: ignore
                    )
                )
                zoo[opponent_idx].update_result(
                    env.get_game_result(
                        env.env.rewards[env.players[1 - player_pos]]  # type: ignore
                    )
                )
            else:
                mask = observation["action_mask"]
                cur_method = (
                    methods[method_idx]
                    if agent == env.players[player_pos]
                    else opponent
                )
                state = env.obs_to_state(
                    observation["observation"],
                    player_pos,
                    obs_mode=cur_method.method.obs_mode,
                )
                action = cur_method.method.act(state, step, mask)
                state_dict[agent] = (state, action, mask)

            env.env.step(action)

            _, reward, _, _, _ = env.env.last()  # type: ignore
            if (
                env.env.agent_selection == env.players[player_pos]  # type: ignore
                and env.env.agent_selection in state_dict  # type: ignore
            ):
                s, a, mask = state_dict[env.players[player_pos]]

                episode.append(ReplayItem(s, a, float(reward), mask))

                methods[method_idx].method.update(episode, step)

        methods[method_idx].method.finalize(episode, step)

        if plot_interval and step % plot_interval == 0 and step > 0:
            for idx, method in enumerate(methods):
                win_ratios[idx].append((step, method.get_win_ratio()))

            plt.clf()
            for idx, method in enumerate(methods):
                x = [epoch for epoch, _ in win_ratios[idx]]
                y = [win_ratio for _, win_ratio in win_ratios[idx]]
                plt.plot(x, y, label=method.method.get_name())

            plt.legend()
            plt.xlabel("Step")
            plt.ylabel("Win %")
            plt.savefig("wins.png")

            log_methods(methods, step)

            for method in methods:
                method.method.save_weights()

        if step % zoo_update_interval == 0:
            zoo.append(methods[method_idx].clone())
            zoo = sorted(zoo, key=lambda x: -x.get_win_ratio())
            zoo = zoo[:zoo_size]

        env.env.close()

