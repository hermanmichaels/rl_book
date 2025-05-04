from dataclasses import dataclass

import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import Discrete


@dataclass
class ParametrizedEnv:
    env: Env
    gamma: float
    _eps_end: float = 0.05
    _eps_start: float = 1
    _num_decay_steps: int = 1000
    intermediate_rewards: bool = False
    eps_decay: bool = False

    def eps(self, step: int) -> float:
        """Returns exploration factor depending on current step.

        Args:
            step: current step

        Returns:
            - constant value if no exploration decay
            - otherwise linearly decaying value
        """
        return (
            self._eps_end
            if not self.eps_decay
            else max(
                self._eps_end,
                self._eps_start
                - step * (self._eps_start - self._eps_end) / self._num_decay_steps,
            )
        )

    def normalized_grid_position_sum(self, observation: int) -> float:
        """Computes the normalized row / column index of the passed observation.
        Used for reward heuristics under the assumption that a higher such
        value is better / closer to the goal.
        """
        assert isinstance(self.env.observation_space, Discrete)
        observation_space: Discrete = self.env.observation_space
        grid_size = np.sqrt(observation_space.n)
        return (observation // grid_size + observation % grid_size) / grid_size

    def step(self, action: int, old_obs: int) -> tuple[int, float, bool, bool, dict]:
        """Executes a step in the environment and, among others, returns new observation
        and observed reward.
        When "intermediate_rewards" is set, augment the reward by a progress heuristic,
        which computes the normalized difference in row / column indices between
        old and new position.

        Args:
            action: action to take
            old_obs: old observation

        Returns:
            - new observation
            - observed reward
            - indicator flags for terminated / truncatad
            - dictionary with additional info
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        if self.intermediate_rewards:
            reward += self.normalized_grid_position_sum(
                observation
            ) - self.normalized_grid_position_sum(old_obs)
        return observation, reward, terminated, truncated, info
