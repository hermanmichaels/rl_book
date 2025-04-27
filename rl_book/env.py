from dataclasses import dataclass

import numpy as np
from gymnasium.core import Env


@dataclass
class ParametrizedEnv:
    env: Env
    gamma: float
    _eps_end: float = 0.05
    _eps_start: float = 1
    _num_decay_steps: int = 1000
    intermediate_rewards: bool = False
    eps_decay: bool = False

    def eps(self, step) -> float:
        return (
            self._eps_end
            if not self.eps_decay
            else max(
                self._eps_end,
                self._eps_start
                - step * (self._eps_start - self._eps_end) / self._num_decay_steps,
            )
        )

    # TODO: name?
    def manhatten_dist(self, observation):
        grid_size = np.sqrt(self.env.observation_space.n)
        return (observation // grid_size + observation % grid_size) / grid_size

    def step(self, action, old_obs):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.intermediate_rewards:
            reward += self.manhatten_dist(observation) - self.manhatten_dist(old_obs)
        return observation, reward, terminated, truncated, info
