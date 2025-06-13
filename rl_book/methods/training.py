# done for unit test?
import random

import matplotlib.pyplot as plt

from rl_book.gym_utils import get_observation_action_space
from rl_book.replay_utils import ReplayItem


def train_single_player(
    env, method, max_steps=100, callback=None
):  # todo: proper empty callback # todo: propery of method class?
    _, action_space = get_observation_action_space(env)
    all_valid_mask = [1 for _ in range(action_space.n)]  # TODO: optional?

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        cur_episode_len = 0
        episode = []

        while not terminated and not truncated:
            action = method.act(observation, all_valid_mask, step)

            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            episode.append(ReplayItem(observation, action, reward, all_valid_mask))
            method.update(episode, step)

            observation = observation_new

            cur_episode_len += 1

        episode.append(ReplayItem(observation_new, -1, reward, []))  # why? sarsa?
        method.finalize(episode, step)

        if callback and callback(method.get_policy(), step):
            return True, method.get_policy(), step

    pi = method.get_policy()

    return False, pi, step


# TODO: only 2 players
def train_multi_player(env, methods, zoo, max_steps: int = 100):
    win_ratios = [[(0, method.get_win_ratio())] for method in methods]

    for step in range(max_steps):
        env.env.reset()

        method_idx = random.randint(0, len(methods) - 1)
        player_pos = random.randint(0, 1)

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
                    action = methods[method_idx].method.act(state, mask, step)
                else:
                    action = opponent.method.act(state, mask, step)
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

        if step % 100 == 0:
            for idx, method in enumerate(methods):
                win_ratios[idx].append((step, method.get_win_ratio()))

            plt.clf()
            for idx in range(len(methods)):
                x = [epoch for epoch, _ in win_ratios[idx]]
                y = [win_ratio for _, win_ratio in win_ratios[idx]]
                plt.plot(x, y, label=idx)

            plt.legend()
            plt.savefig("wins.png")

        if step % 50 == 0:
            zoo.append(methods[method_idx].clone())
            zoo = sorted(zoo, key=lambda x: -x.get_win_ratio())
            zoo = zoo[:50]

        env.env.close()
