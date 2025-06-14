import copy
import random
from collections import defaultdict

import numpy as np

from rl_book.methods.method import RLMethod

ALPHA = 0.1


class TDMethod(RLMethod):
    def __init__(self, env):
        super().__init__(env)
        self.Q = defaultdict(float)

    def clone(self):
        cloned = self.__class__(self.env)
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    def act(self, state, step, mask=None):
        allowed_actions = self.get_allowed_actions(mask)
        if random.uniform(0, 1) < self.env.eps(step):
            return random.choice(allowed_actions)
        else:
            q_values = [self.Q[state, a] for a in allowed_actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(allowed_actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def get_policy(self):
        return np.array(
            [
                np.argmax([self.Q[s, a] for a in range(self.env.env.action_space.n)])
                for s in range(self.env.env.observation_space.n)
            ]
        )


class Sarsa(TDMethod):
    def get_name(self) -> str:
        return "Sarsa"

    def update(self, episode, step):
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        # TODO :action?

        if cur_state.mask is None or np.sum(cur_state.mask) > 0:
            action_new = self.act(cur_state.state, step, cur_state.mask)
            q_next = self.Q[cur_state.state, action_new]
        else:
            q_next = 0

        self.Q[prev_state.state, prev_state.action] = self.Q[
            prev_state.state, prev_state.action
        ] + ALPHA * (
            float(prev_state.reward)
            + self.env.gamma * q_next
            - self.Q[prev_state.state, prev_state.action]
        )

    def finalize(self, episode, step):
        self.update(episode, step)


class QLearning(TDMethod):
    def get_name(self) -> str:
        return "QLearning"

    def update(self, episode, step):
        if len(episode) <= 1:
            return

        # prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]
        allowed_actions = self.get_allowed_actions(cur_state.mask)
        next_q = max(
            [self.Q[next_state.state, a_] for a_ in allowed_actions],
            default=0,
        )  # TODO: maks # tood: right mask index?
        self.Q[cur_state.state, cur_state.action] = self.Q[
            cur_state.state, cur_state.action
        ] + ALPHA * (
            cur_state.reward
            + self.env.gamma * next_q
            - self.Q[cur_state.state, cur_state.action]
        )

    def finalize(self, episode, step):
        self.update(episode, step)


class ExpectedSarsa(TDMethod):
    def get_name(self) -> str:
        return "ExpectedSarsa"

    def _get_action_prob(self, observation_new, a) -> float:
        return (
            self.Q[observation_new, a]
            / sum(
                [
                    self.Q[observation_new, a_]
                    for a_ in range(self.env.get_action_space_len())
                ]
            )
            if sum(
                [
                    self.Q[observation_new, a_]
                    for a_ in range(self.env.get_action_space_len())
                ]
            )
            else 1
        )

    def update(self, episode, step):
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        updated_q_value = self.Q[prev_state.state, prev_state.action] + ALPHA * (
            prev_state.reward - self.Q[prev_state.state, prev_state.action]
        )

        for a in range(self.env.get_action_space_len()):
            updated_q_value += (
                ALPHA
                * self._get_action_prob(cur_state.state, a)
                * self.Q[cur_state.state, a]
            )

        self.Q[prev_state.state, prev_state.action] = updated_q_value

    def finalize(self, episode, step):
        self.update(episode, step)


class DoubleQ(TDMethod):
    def get_name(self) -> str:
        return "DoubleQ"

    def __init__(self, env):
        super().__init__(env)
        self.Q_2 = defaultdict(float)

    def update(self, episode, step):
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]

        # TODO: allowed actions
        if random.randint(0, 100) < 50:
            max_q = max(
                [self.Q[next_state.state, a_] for a_ in range(len(cur_state.mask))],
                default=0,
            )  # TODO: maks # tood: right mask index?
            max_q_a = [
                a_
                for a_ in range(len(cur_state.mask))
                if self.Q[next_state.state, a_] == max_q
            ][0]
            self.Q[cur_state.state, cur_state.action] = self.Q[
                cur_state.state, cur_state.action
            ] + ALPHA * (
                cur_state.reward
                + self.env.gamma * self.Q_2[next_state.state, max_q_a]
                - self.Q[cur_state.state, cur_state.action]
            )
        else:
            max_q = max(
                [self.Q_2[next_state.state, a_] for a_ in range(len(cur_state.mask))],
                default=0,
            )  # TODO: maks # tood: right mask index?
            max_q_a = [
                a_
                for a_ in range(len(cur_state.mask))
                if self.Q_2[next_state.state, a_] == max_q
            ][0]
            self.Q_2[cur_state.state, cur_state.action] = self.Q_2[
                cur_state.state, cur_state.action
            ] + ALPHA * (
                cur_state.reward
                + self.env.gamma * self.Q[next_state.state, max_q_a]
                - self.Q_2[cur_state.state, cur_state.action]
            )

    def finalize(self, episode, step):
        self.update(episode, step)
