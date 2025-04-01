from dataclasses import dataclass

from gymnasium.core import Env


@dataclass
class ParametrizedEnv:
    env: Env
    gamma: float
    eps: float
