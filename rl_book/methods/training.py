# done for unit test?
import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import Env

from rl_book.methods.method import MethodWithStats, RLMethod
from rl_book.replay_utils import ReplayItem


def train_single_player(
    env: Env, method: RLMethod, max_steps: int = 100, callback: Callable | None = None
) -> tuple[bool, np.ndarray, int]:
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

        while not terminated and not truncated:
            action = method.act(observation, step)

            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            episode.append(ReplayItem(observation, action, reward))
            method.update(episode, step)

            observation = observation_new

        episode.append(ReplayItem(observation_new, -1, reward, []))  # why? sarsa?
        method.finalize(episode, step)

        if callback and callback(method.get_policy(), step):
            return True, method.get_policy(), step
        
    env.env.close()

    # TOOD: need to return policy?
    return False, method.get_policy(), step


def train_multi_player(
    env: Env,
    methods: list[MethodWithStats],
    zoo: list[RLMethod],
    max_steps: int = 100,
    zoo_update_interval: int = 50,
    zoo_size: int = 50,
    plot_interval: int = None,
) -> None:
    """Trains a method on multi-player environments (atm only 2 players are supported).

    Args:
        env: env to use
        methods: methods to train
        zoo: initial list of opponents
        max_steps: maixmal number of update steps
    """
    # For plotting: keep (step, win_ratio) tuples for every method at different steps.
    win_ratios = [[(0, method.get_win_ratio())] for method in methods]

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
            agent = env.env.agent_selection
            observation, reward, termination, truncation, info = env.env.last()
            reward += 0.1  # todo

            done = termination or truncation

            if done:
                action = None

                # Game over, rewards contains all
                if env.env.rewards[env.players[player_pos]] == 1:
                    methods[method_idx].update_win()
                elif env.env.rewards[env.players[1 - player_pos]] == 1:
                    zoo[opponent_idx].update_win()

            else:
                mask = observation["action_mask"]
                state = env.obs_to_state(observation["observation"], player_pos)
                if agent == env.players[player_pos]:
                    action = methods[method_idx].method.act(state, step, mask)
                else:
                    action = opponent.method.act(state, step, mask)
                state_dict[agent] = (state, action, mask)

            env.env.step(action)

            _, reward, _, _, _ = env.env.last()

            if (
                env.env.agent_selection == env.players[player_pos]
                and env.env.agent_selection in state_dict
            ):
                s, a, mask = state_dict[env.players[player_pos]]

                observation_new = env.env.observe(env.players[player_pos])
                mask_new = observation_new["action_mask"]

                episode.append(ReplayItem(s, a, reward, mask))

                if reward == 0:
                    legal = []
                else:
                    legal = observation_new["action_mask"]

                methods[method_idx].method.update(episode, step)

        episode.append(
            ReplayItem(
                env.obs_to_state(observation_new["observation"], player_pos), -1, 0, []
            )
        )

        methods[method_idx].method.finalize(episode, step)

        if plot_interval and step % plot_interval == 0:
            for idx, method in enumerate(methods):
                win_ratios[idx].append((step, method.get_win_ratio()))

            plt.clf()
            for idx, method in enumerate(methods):
                x = [epoch for epoch, _ in win_ratios[idx]]
                y = [win_ratio for _, win_ratio in win_ratios[idx]]
                plt.plot(x, y, label=method.method.get_name())

            plt.legend()
            plt.savefig("wins.png")

        if step % zoo_update_interval == 0:
            zoo.append(methods[method_idx].clone())
            zoo = sorted(zoo, key=lambda x: -x.get_win_ratio())
            zoo = zoo[:zoo_size]

        env.env.close()
