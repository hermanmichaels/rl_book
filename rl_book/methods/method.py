import os
import pickle
from abc import ABC
from typing import Any

import numpy as np

from rl_book.env import GameResult, ParametrizedEnv
from rl_book.replay_utils import ReplayItem

SAVE_PATH = "weights/"


class RLMethod(ABC):
    """Base class for RL methods."""

    def __init__(self, env: ParametrizedEnv, load_weights: bool = False) -> None:
        self.env = env
        self._train = True

        if load_weights:
            self.load_weights()

    def get_name(self) -> str:
        """Returns the method's name.

        Returns:
            method's name
        """
        raise NotImplementedError

    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | list = []
    ) -> int:
        """Called during training to act when generating episodes.

        Args:
            state: current state
            mask: mask of allowed actions
            step: current step

        Returns:
            selected action
        """
        raise NotImplementedError

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

    def clone(self):
        cloned = self.__class__(self.env)
        return cloned

    def get_allowed_actions(self, mask: np.ndarray | list) -> np.ndarray:
        """Gets the allowed action indices.

        Args:
            mask: mask of allowed actions (e.g. [1 1 0 0 1 ... ])

        Returns:
            indices of allowed actions (e.g. [0, 1, 4, ...])
        """
        return (
            np.nonzero(mask)[0].tolist()
            if mask is not None
            else np.asarray([a for a in range(self.env.get_action_space_len())])
        )

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def load_weights(self) -> None:
        save_path = SAVE_PATH + self.get_name() + ".pkl"
        if os.path.exists(save_path):
            self._load_weights(save_path)

    def _load_weights(self, save_path: str) -> None:
        raise NotImplementedError

    def save_weights(self) -> None:
        data = self._get_save_data()

        if data:
            save_path = SAVE_PATH + self.get_name() + ".pkl"
            os.makedirs(SAVE_PATH, exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(data, f)

    def _get_save_data(self) -> Any:
        raise NotImplementedError


class MethodWithStats:
    """Wrapper around RLMethod which keeps track of win / lose stats for
    multi-player games."""

    def __init__(self, method: RLMethod) -> None:
        self.method = method
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.picks = 0

    def update_pick(self) -> None:
        self.picks += 1

    def update_result(self, result: GameResult) -> None:
        if result == GameResult.WIN:
            self.wins += 1
        elif result == GameResult.DRAW:
            self.draws += 1
        elif result == GameResult.LOSS:
            self.losses += 1
        else:
            print(f"Unknown game result {result}")

    def get_win_ratio(self):
        return self.wins / (self.picks + 1)

    def get_draw_ratio(self):
        return self.draws / (self.picks + 1)

    def get_loss_ratio(self):
        return self.losses / (self.picks + 1)

    def clone(self):
        cloned = MethodWithStats(self.method.clone())
        cloned.wins = self.wins
        cloned.picks = self.picks
        return cloned
