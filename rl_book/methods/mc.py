import copy
from collections import defaultdict

import numpy as np

from rl_book.methods.method import Algorithm

EPS = 1e-1


def get_eps_greedy_policy(Q, step):
    # env
    QQ = np.zeros((16, 4))
    for i in range(16):
        for j in range(4):
            QQ[i, j] = Q[i, j]
    b = np.zeros_like(QQ) + 0.05 / 4
    optimal_actions = np.argmax(QQ, 1)
    b[np.arange(QQ.shape[0]), optimal_actions] = (
        b[np.arange(QQ.shape[0]), optimal_actions] + 1 - 0.05
    )
    return b


class MCMethod(Algorithm):
    def __init__(self, env):
        super().__init__(env)
        self.Q = defaultdict(float)
        # TODO: misused
        self.pi = (
            np.ones(
                (self.env.get_observation_space_len(), self.env.get_action_space_len())
            ).astype(np.int32)
            / self.env.get_action_space_len()
        )

    def clone(self):
        cloned = super().clone()
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    def act(self, state, mask, step):
        # TOOD: optional?
        actions = range(self.env.get_action_space_len())
        return np.random.choice(actions, p=self.pi[state])

    # TODO: eps greedy
    def get_policy(self):
        return np.array(
            [
                np.argmax(
                    [self.Q[s, a] for a in range(self.env.get_action_space_len())]
                )
                for s in range(self.env.get_observation_space_len())
            ]
        )


class OnPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.counts = defaultdict(int)

    def finalize(self, episode, step):
        G = 0.0
        for t in range(len(episode) - 2, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward

            G = self.env.gamma * G + r
            prev_s = [(item.state, item.action) for item in episode[:t]]
            if (s, a) not in prev_s:
                self.counts[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.counts[s, a]

                if not all(
                    self.Q[s, a] == self.Q[s, 0]
                    for a in range(self.env.env.action_space.n)
                ):
                    A_star = np.argmax(
                        [self.Q[s, a] for a in range(self.env.env.action_space.n)]
                    )
                    for a in range(self.env.env.action_space.n):
                        self.pi[s, a] = (
                            1
                            - self.env.eps(step)
                            + self.env.eps(step) / self.env.env.action_space.n
                            if a == A_star
                            else self.env.eps(step) / self.env.env.action_space.n
                        )


class OffPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.C = defaultdict(int)

    def get_greedy_policy(self):
        return np.argmax(
            np.asarray(
                [
                    [self.Q[s, a] for a in range(self.env.env.action_space.n)]
                    for s in range(self.env.env.observation_space.n)
                ]
            ),
            1,
        )

    def get_eps_greedy_policy(self, step):
        # TODO: speedup
        Q = np.asarray(
            [
                [self.Q[state, a] for a in range(self.env.env.action_space.n)]
                for state in range(self.env.env.observation_space.n)
            ]
        )
        pi = np.zeros_like(Q) + self.env.eps(step) / self.env.env.action_space.n
        optimal_actions = np.argmax(Q, 1)
        pi[np.arange(Q.shape[0]), optimal_actions] += 1 - self.env.eps(step)
        return pi

    def finalize(self, episode, step):
        # TODO: "randomly" can reuse act because eps-greedy policy, but could be any other
        pi_target = self.get_greedy_policy()

        # import ipdb
        # ipdb.set_trace()

        G = 0.0
        W = 1
        for t in range(len(episode) - 2, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            G = self.env.gamma * G + r
            self.C[s, a] += W
            self.Q[s, a] += W / self.C[s, a] * (G - self.Q[s, a])
            pi_target = self.get_greedy_policy()
            if a != pi_target[s]:
                break
            W *= 1 / self.pi[s, a]

        self.pi = self.get_eps_greedy_policy(step)

        # TODO: return pi_target eventually


class OffPolicyMCNonInc(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.returns = defaultdict(list)
        self.ratios = defaultdict(list)

    def get_eps_greedy_policy(self, step):
        # TODO: speedup
        Q = np.asarray(
            [
                [self.Q[state, a] for a in range(self.env.env.action_space.n)]
                for state in range(self.env.env.observation_space.n)
            ]
        )
        pi = np.zeros_like(Q) + self.env.eps(step) / self.env.env.action_space.n
        optimal_actions = np.argmax(Q, 1)
        pi[np.arange(Q.shape[0]), optimal_actions] += 1 - self.env.eps(step)
        return pi

    def finalize(self, episode, step):
        G = 0.0
        ratio = 1

        # Create current target policy,
        # which is the argmax of the Q function,
        # but gives equal weighting to tied Q values.
        Q = np.asarray(
            [
                [self.Q[state, a] for a in range(self.env.env.action_space.n)]
                for state in range(self.env.env.observation_space.n)
            ]
        )
        pi = np.zeros_like(Q)
        optimal_actions = np.argmax(Q, 1)
        pi[np.arange(Q.shape[0]), optimal_actions] = 1
        uniform_rows = np.all(Q == Q[:, [0]], axis=1)
        pi[uniform_rows] = 1 / self.env.env.action_space.n

        for t in range(len(episode) - 2, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            G = self.env.gamma * G + r

            ratio *= pi[s, a] / self.pi[s, a]

            if ratio == 0:
                break

            self.returns[s, a].append(G)
            self.ratios[s, a].append(ratio)

            self.Q[s, a] = sum(
                [r * s for r, s in zip(self.returns[s, a], self.ratios[s, a])]
            ) / sum([s for s in self.ratios[s, a]])

        self.pi = self.get_eps_greedy_policy(step)
