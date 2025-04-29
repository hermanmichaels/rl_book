import copy
import random
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method_wrapper import with_default_values

_cached_functions = []


def call_once(func):
    """Custom cache decorator
    treating the first argument (env)
    in a special way (key is env.n-{args[1:]}).
    """
    cache = {}

    def wrapper(*args, **kwargs):
        assert isinstance(args[0], ParametrizedEnv), ""

        key = (func.__name__, args[0].env.observation_space.n, args[1:])
        assert not kwargs, "We don't support kwargs atm"
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]

    def clear_cache():
        cache.clear()

    wrapper.clear_cache = clear_cache
    _cached_functions.append(wrapper)

    return wrapper


def clear_all_caches():
    for func in _cached_functions:
        func.clear_cache()


@call_once
def generate_possible_states(
    env: ParametrizedEnv, num_runs: int = 10000
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
    env: ParametrizedEnv, pi: np.ndarray, exploring_starts: bool
) -> list[tuple[Any, Any, Any]]:
    """Generate an episode following the given policy.

    Args:
        env: environment to use
        pi: policy to follow
        exploring_starts: true when to follow exploring state assumption (ES)
        max_episode_length: TODO

    Returns:
        generated episode
    """
    observation_space, action_space = get_observation_action_space(env)

    max_episode_length = observation_space.n * 4

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

        observation_new, reward, terminated, truncated, _ = env.step(
            action, observation
        )

        episode.append((observation, action, reward))

        # Terminate episodes in which agent is likely stuck
        if len(episode) > max_episode_length:
            break

        observation = observation_new

    return episode


def get_eps_greedy_policy(env, Q, step):
    b = np.ones_like(Q) * (env.eps(step) / Q.shape[1])
    optimal_actions = np.argmax(Q, axis=1)
    b[np.arange(Q.shape[0]), optimal_actions] += 1 - env.eps(step)
    return b


@with_default_values
def mc_es(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
) -> tuple[bool, np.ndarray, int]:
    """Solve passed Gymnasium env via Monte Carlo
    with exploring starts.

    Args:
        env: env containing the problem

    Returns:
        found policy
    """
    clear_all_caches()

    observation_space, action_space = get_observation_action_space(env)

    pi = (
        np.ones((observation_space.n, action_space.n)).astype(np.int32) / action_space.n
    )
    Q = np.zeros((observation_space.n, action_space.n))
    counts = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        episode = generate_episode(env, pi, True)
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = env.gamma * G + r
            prev_s = [(s, a) for (s, a, _) in episode[:t]]
            if (s, a) not in prev_s:
                counts[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / counts[s, a]
                if not all(Q[s, a] == Q[s, 0] for a in range(action_space.n)):
                    for a in range(action_space.n):
                        pi[s, a] = 1 if a == np.argmax(Q[s]) else 0

        p = np.argmax(pi, 1)
        if success_cb(p, step):
            return True, p, step

    return False, np.argmax(pi, 1), step


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


@with_default_values
def off_policy_mc_non_inc(
    env: ParametrizedEnv,
    success_cb: Callable[[np.ndarray, int], bool],
    max_steps: int,
    eps_decay: bool = False,
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
