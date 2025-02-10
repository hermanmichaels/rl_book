from dataclasses import dataclass
import numpy as np

from gymnasium.core import Env


@dataclass
class ParametrizedEnv:
    env: Env
    gamma: float
    eps: float # TODO
    intermediate_rewards: bool = False

    def manhatten_dist(self, observation):
        grid_size = np.sqrt(self.env.observation_space.n)
        return (observation // grid_size + observation % grid_size) / grid_size

    def step(self, action, old_obs):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.intermediate_rewards:
            reward += self.manhatten_dist(observation) - self.manhatten_dist(old_obs)
        return observation, reward, terminated, truncated, info
