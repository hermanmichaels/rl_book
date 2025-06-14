from abc import ABC

import numpy as np
from gymnasium.core import Env  # TODO: or any other env

from rl_book.replay_utils import ReplayItem


class RLMethod(ABC):
    """Base class for RL methods."""

    def __init__(self, env: Env) -> None:
        self.env = env

    def get_name() -> str:
        """Returns the method's name.

        Returns:
            method's name
        """
        raise NotImplementedError

    def act(self, state, step: int, mask: list[int] | None = None) -> int:
        """Called during training to act when generating episodes.

        Args:
            state: current state
            mask: mask of allowed actions
            step: current step

        Returns:
            selected action
        """
        pass

    def update(self, episode: list[ReplayItem], step: int) -> None:
        """Updates the method's parameters.

        Args:
            episode: episode generated up to now
            step: current step
        """
        pass

    def finalize(self, episode: list[ReplayItem], step: int) -> None:
        """Called when one episode generation has finished.

        Args:
            episode: complete episode
            step: current step
        """
        pass

    def get_policy(self) -> np.ndarray:
        """Returns the policy found by this method.

        Returns:
            Found policy
        """
        raise NotImplementedError

    def clone(self):
        cloned = self.__class__(self.env)
        return cloned

    def get_allowed_actions(self, mask: np.ndarray) -> np.ndarray:
        """Gets the allowed action indices.

        Args:
            mask: mask of allowed actions (e.g. [1 1 0 0 1 ... ])

        Returns:
            indices of allowed actions (e.g. [0, 1, 4, ...])
        """
        return (
            np.nonzero(mask)[0].tolist()
            if mask is not None
            else [1 for _ in range(self.env.get_action_space_len())]
        )


class MethodWithStats:
    """Wrapper around RLMethod which keeps track of win / lose stats for
    multi-player games."""

    def __init__(self, method: RLMethod) -> None:
        self.method = method
        self.wins = 0
        self.picks = 0

    def update_pick(self) -> None:
        self.picks += 1

    def update_win(
        self,
    ) -> None:
        self.wins += 1

    def get_win_ratio(self):
        return self.wins / (self.picks + 1)

    def clone(self):
        cloned = MethodWithStats(self.method.clone())
        cloned.wins = self.wins
        cloned.picks = self.picks
        return cloned
