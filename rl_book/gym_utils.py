from gymnasium.spaces import Discrete

from rl_book.env import ParametrizedEnv


def get_observation_action_space(env: ParametrizedEnv) -> tuple[Discrete, Discrete]:
    """Extracts observation and action space from given environment.
    This function solely exists to satisfy mypy type hints.
    """
    assert isinstance(env.env.observation_space, Discrete)
    observation_space: Discrete = env.env.observation_space
    assert isinstance(env.env.action_space, Discrete)
    action_space: Discrete = env.env.action_space

    return observation_space, action_space
