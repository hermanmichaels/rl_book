import copy
import random
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method import Algorithm
from rl_book.methods.method_wrapper import with_default_values
from rl_book.methods.td import TDMethod

EPS = 1e-1


def get_eps_greedy_policy(Q, step):
    # env
    QQ = np.zeros((16, 4))
    for i in range(16):
        for j in range(4):
            QQ[i, j] = Q[i, j]
    b = np.zeros_like(QQ) + 0.05 / 4
    optimal_actions = np.argmax(QQ, 1)
    b[np.arange(QQ.shape[0]), optimal_actions] = b[np.arange(QQ.shape[0]), optimal_actions] + 1 - 0.05
    # import ipdb
    # ipdb.set_trace()
    return b

class MCMethod(Algorithm):
    def __init__(self, env):
        super().__init__(env)
        self.Q = defaultdict(float)

    def clone(self):
        cloned = super().clone()
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

    def get_eps_greedy_policy(self, step):
        # TODO: speedup
        Q = np.asarray([[self.Q[state, a] for a in range(self.env.env.action_space.n)] for state in range(self.env.env.observation_space.n)])
        b = np.zeros_like(Q) + self.env.eps(step) / self.env.env.action_space.n
        optimal_actions = np.argmax(Q, 1)
        b[np.arange(Q.shape[0]), optimal_actions] = b[np.arange(Q.shape[0]), optimal_actions] + 1 - self.env.eps(step)
        return b
    
    def act(self, state, mask, step):
        actions = range(self.env.env.action_space.n)
        b = self.get_eps_greedy_policy(step)
        return np.random.choice(actions, p=b[state])

class OnPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.counts = defaultdict(int)

    def finalize(self, episode, step):
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward

            G = self.env.gamma * G + r
            prev_s = [(item.state, item.action) for item in episode[:t]]
            if (s, a) not in prev_s:
                self.counts[s, a] += 1
                self.Q[s, a] += (G - self.Q[s, a]) / self.counts[s, a]


@with_default_values
def on_policy_mc(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
) -> tuple[bool, np.ndarray, int]:
    """Solve passed Gymnasium env via on-policy Monte
    Carlo control.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    pi = (
        np.ones((observation_space.n, action_space.n)).astype(np.int32) / action_space.n
    )
    Q = np.zeros((observation_space.n, action_space.n))
    counts = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        episode = generate_episode(env, pi, False)

        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r
            prev_s = [(s, a) for (s, a, _) in episode[:t]]
            if (s, a) not in prev_s:
                counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / counts[s, a]
                if not all(Q[s, a] == Q[s, 0] for a in range(action_space.n)):
                    A_star = np.argmax(Q[s, :])
                    for a in range(action_space.n):
                        pi[s, a] = (
                            1 - env.eps(step) + env.eps(step) / action_space.n
                            if a == A_star
                            else env.eps(step) / action_space.n
                        )

        p = np.argmax(pi, 1)
        if success_cb(p, step):
            return True, p, step

    return False, np.argmax(pi, 1), step

class OffPolicyMC(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.C = defaultdict(int)

    def update(self, episode):
        pass

    def finalize(self, episode, step):
        # TODO: "randomly" can reuse act because eps-greedy policy, but could be any other
        b = self.get_eps_greedy_policy(step) # todo: not quite right ...

        G = 0.0
        W = 1
        for t in range(len(episode) - 1, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            G = self.env.gamma * G + r
            self.C[s, a] += W
            self.Q[s, a] += W / self.C[s, a] * (G - self.Q[s, a])
            if a != np.argmax([self.Q[s, a] for a in range(4)]):
                break
            W *= 1 / b[s, a]

@with_default_values
def off_policy_mc(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
) -> tuple[bool, np.ndarray, int]:
    """Solve passed Gymnasium env via off-policy Monte
    Carlo control.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    Q = np.zeros((observation_space.n, action_space.n))
    C = np.zeros((observation_space.n, action_space.n))
    pi = np.argmax(Q, 1)

    for step in range(max_steps):
        b = get_eps_greedy_policy(env, Q, step)
        episode = generate_episode(env, b, False)

        G = 0.0
        W = 1
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r
            C[s, a] += W
            Q[s, a] += W / C[s, a] * (G - Q[s, a])
            pi = np.argmax(Q, 1)
            if a != np.argmax(Q[s]):
                break
            W *= 1 / b[s, a]

        if success_cb(pi, step):
            return True, pi, step

    return False, pi, step

class OffPolicyMCNonInc(MCMethod):
    def __init__(self, env):
        super().__init__(env)
        self.returns = defaultdict(list)
        self.ratios = defaultdict(list)

    def finalize(self, episode, step):
        b = self.get_eps_greedy_policy(step)

        G = 0.0
        ratio = 1
        for t in range(len(episode) - 1, -1, -1):
            s = episode[t].state
            a = episode[t].action
            r = episode[t].reward
            G = 0.95 * G + r

            # # Create current target policy,
            # # which is the argmax of the Q function,
            # # but gives equal weighting to tied Q values
            # pi = np.zeros_like(self.Q)
            # pi[np.arange(16), np.argmax(self.Q, 1)] = 1
            # uniform_rows = np.all(self.Q == self.Q[:, [0]], axis=1)
            # pi[uniform_rows] = 1 / 4

            Q = np.asarray([[self.Q[state, a] for a in range(self.env.env.action_space.n)] for state in range(self.env.env.observation_space.n)])
            pi = np.zeros_like(Q)
            optimal_actions = np.argmax(Q, 1)
            pi[np.arange(Q.shape[0]), optimal_actions] = 1

            ratio *= pi[s, a] / b[s, a]
            if ratio == 0:
                break

            self.returns[s, a].append(G)
            self.ratios[s, a].append(ratio)

            self.Q[s, a] = sum([r * s for r, s in zip(self.returns[s, a], self.ratios[s, a])]) / sum(
                [s for s in self.ratios[s, a]]
            )

@with_default_values
def off_policy_mc_non_inc(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
) -> tuple[bool, np.ndarray, int]:
    """Solve passed Gymnasium env via on-policy Monte
    Carlo control - but does not use incremental algorithm
    from Sutton for updating the importance sampling weights.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    observation_space, action_space = get_observation_action_space(env)

    Q = np.zeros((observation_space.n, action_space.n))

    returns = defaultdict(list)
    ratios = defaultdict(list)

    for step in range(max_steps):
        b = get_eps_greedy_policy(env, Q, step)
        episode = generate_episode(env, b, False)

        G = 0.0
        ratio = 1
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r

            # Create current target policy,
            # which is the argmax of the Q function,
            # but gives equal weighting to tied Q values
            pi = np.zeros_like(Q)
            pi[np.arange(Q.shape[0]), np.argmax(Q, 1)] = 1
            uniform_rows = np.all(Q == Q[:, [0]], axis=1)
            pi[uniform_rows] = 1 / action_space.n

            ratio *= pi[s, a] / b[s, a]
            if ratio == 0:
                break

            returns[s, a].append(G)
            ratios[s, a].append(ratio)

            Q[s, a] = sum([r * s for r, s in zip(returns[s, a], ratios[s, a])]) / sum(
                [s for s in ratios[s, a]]
            )

        p = np.argmax(Q, 1)
        if success_cb(p, step):
            return True, p, step

    Q = np.nan_to_num(Q, nan=0.0)
    return False, np.argmax(Q, 1), step
