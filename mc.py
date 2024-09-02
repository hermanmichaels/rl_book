import copy
import random
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from env import ParametrizedEnv
from gym_utils import get_observation_action_space


def call_once(func):
    """Custom cache decorator
    ignoring the first argument.
    """
    cache = {}

    def wrapper(*args, **kwargs):
        key = (func.__name__, args[1:])
        assert not kwargs, "We don't support kwargs atm"
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]

    return wrapper


@call_once
def generate_possible_states(
    env: ParametrizedEnv, num_runs: int = 100
) -> list[tuple[int, ParametrizedEnv]]:
    """Generate possible states of an environment.
    For this, iteratively increase the set of already known
    states by picking a random state and following a random
    policy from then on, noting down new states.

    Args:
        env: environemnt to use
        num_runs: number of discover loops

    Returns:
        list containing found states - which are tuples of states (observations)
        and the gym environment reprsenting that state
    """
    _, action_space = get_observation_action_space(env)

    observation, _ = env.env.reset()
    possible_states = [(observation, copy.deepcopy(env))]

    for _ in range(num_runs):
        observation, env = random.choice(possible_states)
        env = copy.deepcopy(env)
        terminated = truncated = False
        while not terminated and not truncated:
            action = np.random.choice([a for a in range(action_space.n)])
            observation, _, terminated, truncated, _ = env.env.step(action)
            if observation in set([state for state, _ in possible_states]):
                break
            else:
                if not terminated and not truncated:
                    possible_states.append((observation, env))

    return possible_states


def generate_random_start(env: ParametrizedEnv) -> tuple[int, ParametrizedEnv]:
    """Pick a random starting state.
    For that, first generate all possible
    states (cached), then pick a random
    state from these.
    """
    possible_states = generate_possible_states(env)
    observation, env = random.choice(possible_states)
    return copy.deepcopy(observation), copy.deepcopy(env)


def generate_episode(
    env: ParametrizedEnv,
    pi: np.ndarray,
    exploring_starts: bool,
    max_episode_length: int = 20,
) -> list[tuple[Any, Any, Any]]:
    """Generate an episode following the given policy.

    Args:
        env: environment to use
        pi: policy to follow
        exploring_starts: true when to follow exploring state assumption (ES)

    Returns:
        generated episode
    """
    _, action_space = get_observation_action_space(env)

    episode = []

    observation, _ = env.env.reset()

    if exploring_starts:
        # Pick random starting state if ES
        observation, env = generate_random_start(env)

    terminated = truncated = False
    initial_step = True

    while not terminated and not truncated:
        if initial_step and exploring_starts:
            # Pick random action initially if ES
            action = np.random.choice([a for a in range(action_space.n)])
        else:
            action = np.random.choice(
                [a for a in range(action_space.n)], p=pi[observation]
            )
        initial_step = False

        observation_new, reward, terminated, truncated, _ = env.env.step(action)

        episode.append((observation, action, reward))

        # Terminate episodes in which agent is stuck
        if len(episode) > max_episode_length:
            break

        observation = observation_new

    return episode


def mc_es(env: ParametrizedEnv) -> np.ndarray:
    """Solve passed Gymnasium env via Monte Carlo
    with exploring starts.

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

    returns = defaultdict(list)
    num_steps = 1000

    for t in range(num_steps):
        episode = generate_episode(env, pi, True)

        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r
            prev_s = [(s, a) for (s, a, _) in episode[:t]]
            if (s, a) not in prev_s:
                returns[s, a].append(G)
                Q[s, a] = statistics.fmean(returns[s, a])
                if not all(Q[s, a] == Q[s, 0] for a in range(action_space.n)):
                    for a in range(action_space.n):
                        pi[s, a] = 1 if a == np.argmax(Q[s]) else 0

    return np.argmax(pi, 1)


def on_policy_mc(env: ParametrizedEnv) -> np.ndarray:
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

    returns = defaultdict(list)
    num_steps = 1000

    for _ in range(num_steps):
        episode = generate_episode(env, pi, False)

        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r
            prev_s = [(s, a) for (s, a, _) in episode[:t]]
            if (s, a) not in prev_s:
                returns[s, a].append(G)
                Q[s, a] = statistics.fmean(returns[s, a])
                if not all(Q[s, a] == Q[s, 0] for a in range(action_space.n)):
                    A_star = np.argmax(Q[s, :])
                    for a in range(action_space.n):
                        pi[s, a] = (
                            1 - env.eps + env.eps / action_space.n
                            if a == A_star
                            else env.eps / action_space.n
                        )

    return np.argmax(pi, 1)


def off_policy_mc(env: ParametrizedEnv) -> np.ndarray:
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

    num_steps = 1000

    for _ in range(num_steps):
        b = np.random.rand(int(observation_space.n), int(action_space.n))
        b = b / np.expand_dims(np.sum(b, axis=1), -1)

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

    return pi


def off_policy_mc_non_inc(env: ParametrizedEnv) -> np.ndarray:
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

    num_steps = 10000
    returns = defaultdict(list)
    ratios = defaultdict(list)

    for _ in range(num_steps):
        b = np.random.rand(int(observation_space.n), int(action_space.n))
        b = b / np.expand_dims(np.sum(b, axis=1), -1)

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

    Q = np.nan_to_num(Q, nan=0.0)

    return np.argmax(Q, 1)
