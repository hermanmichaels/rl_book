import random
from typing import Any, ClassVar

import numpy as np
import torch

from rl_book.env import ObsMode, ParametrizedEnv
from rl_book.methods.method import RLMethod


class Random(RLMethod):
    def __init__(
        self, env: ParametrizedEnv, device: torch.device = torch.device("cpu")
    ):
        super().__init__(env)

    def get_name(self) -> str:
        return "Random"

    def act(self, state: int, step: int | None = None, mask: np.ndarray | list = []):
        allowed_actions = self.get_allowed_actions(mask)
        return random.choice(allowed_actions)

    def _get_save_data(self) -> Any:
        return None
    
class RandomBatched(RLMethod):
    obs_mode: ClassVar[ObsMode] = ObsMode.RASTERIZED
    
    def __init__(
        self, env: ParametrizedEnv, device: torch.device = torch.device("cpu")
    ):
        super().__init__(env)

    def get_name(self) -> str:
        return "Random"

    def act(self, state: int, step: int | None = None, mask: np.ndarray | list = []):
        allowed_actions = self.get_allowed_actions(mask).float()

        probs = allowed_actions

        row_sum = probs.sum(dim=1)
        invalid_zero_sum = row_sum <= 0

        if invalid_zero_sum.any():
            probs[invalid_zero_sum] += 1 / probs.shape[1]

        if torch.sum(probs) == 0:
            probs += 1 / probs.shape[1]

        return torch.multinomial(probs, num_samples=1).squeeze(1)

    def _get_save_data(self) -> Any:
        return None

