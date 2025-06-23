import copy
from collections import defaultdict
from typing import DefaultDict

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod
from rl_book.replay_utils import ReplayItem


class MCMethod(RLMethod):
    def __init__(self, env: ParametrizedEnv) -> None:
        super().__init__(env)
        self.Q: DefaultDict[tuple[int, int], float] = defaultdict(float)
        self.pi: DefaultDict[tuple[int, int], float] = defaultdict(lambda: 1.0)

    def clone(self):
        cloned = super().clone()
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | None = None
    ) -> int:
        actions = self.get_allowed_actions(mask)
        probs_arr = [self.pi[state, a] for a in actions]

        if self._train:
            probs = np.exp(probs_arr - np.max(probs_arr))
            probs /= sum(probs)
            return np.random.choice(actions, p=probs)
        else:
            return actions[np.argmax(probs_arr)]


class OnPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.counts = defaultdict(int)

    def get_name(self) -> str:
        return "OnPolicyMc"

    def finalize(self, episode: list[ReplayItem], step: int) -> None:
        G = 0.0
        for t in range(len(episode) - 2, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            mask = episode[t].mask

            actions = self.get_allowed_actions(mask)

            G = self.env.gamma * G + r
            prev_s = [(item.state, item.action) for item in episode[:t]]
            if (s, a) not in prev_s:
                self.counts[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.counts[s, a]

                if not all(
                    self.Q[s, a] == self.Q[s, 0]
                    for a in range(self.env.get_action_space_len())
                ):
                    A_star = np.argmax(
                        [
                            self.Q[s, a] if a in actions else self.Q[s, a] - np.inf
                            for a in range(self.env.get_action_space_len())
                        ]
                    )
                    for a in range(self.env.get_action_space_len()):
                        self.pi[s, a] = (
                            1
                            - self.env.eps(step)
                            + self.env.eps(step) / self.env.get_action_space_len()
                            if a == A_star
                            else self.env.eps(step) / self.env.get_action_space_len()
                        )


class OffPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.C = defaultdict(int)

    def get_name(self) -> str:
        return "OffPolicyMC"

    def set_eps_greedy_behavior_policy(self, step: int) -> None:
        n_actions = self.env.get_action_space_len()
        eps = self.env.eps(step)
        seen_states = {s for s, _ in self.Q.keys()}

        for s in seen_states:
            # Build a NumPy array of Q-values for this state
            qs = np.array([self.Q[s, a] for a in range(n_actions)])
            best_action = np.argmax(qs)

            for a in range(n_actions):
                self.pi[s, a] = 1 - eps if a == best_action else eps

    def finalize(self, episode, step):
        # Note: self.pi is here used as the behavior policy b, while the target policy π
        # is implicity represented by argmax(Q).
        G = 0.0
        W = 1
        for t in range(len(episode) - 2, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            mask = episode[t].mask

            actions = self.get_allowed_actions(mask)

            G = self.env.gamma * G + r
            self.C[s, a] += W
            self.Q[s, a] += W / self.C[s, a] * (G - self.Q[s, a])
            next_qs = [
                self.Q[s, a_] if a_ in actions else self.Q[s, a] - np.inf
                for a_ in range(self.env.get_action_space_len())
            ]
            if a != np.argmax(next_qs):
                break
            W *= 1 / (self.pi[s, a])

        # Improve the behavior policy by any kind of greedy policy - in particular
        # here we form an ε-greedy policy from Q.
        self.set_eps_greedy_behavior_policy(step)
