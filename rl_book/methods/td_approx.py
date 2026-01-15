import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_book.env import ParametrizedEnv
from rl_book.methods.method import RLMethod
from rl_book.replay_utils import ReplayItem

ALPHA = 0.1


class ApproximateTDMethod(RLMethod):
    def __init__(self, env: ParametrizedEnv, load_weights: bool = False) -> None:
        super().__init__(env, load_weights)

    def clone(self) -> "ApproximateTDMethod":
        cloned = self.__class__(self.env, False)
        return cloned

    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | list = []
    ) -> int:
        allowed_actions = self.get_allowed_actions(mask)
        if self._train and step and random.uniform(0, 1) < self.env.eps(step):
            return random.choice(allowed_actions)
        else:
            q_values = [self.q(state, a) for a in allowed_actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(allowed_actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def _get_save_data(self) -> Any:
        pass

    def _load_weights(self, save_path: str) -> None:
        pass


class SemiGradientSarsaLinear(ApproximateTDMethod):
    def __init__(self, env: ParametrizedEnv, load_weights: bool = False) -> None:
        super().__init__(env, load_weights)

        self.num_states = env.get_observation_space_len()
        self.num_actions = env.get_action_space_len()

        self.w = np.zeros(self.num_states * self.num_actions)

    def get_name(self) -> str:
        return "Sarsa"

    def feature_fn(self, state, action):
        x = np.zeros(self.num_states * self.num_actions)
        idx = state * self.num_actions + action
        x[idx] = 1.0
        return x

    def q(self, state, action):
        """Approximate action-value."""
        return np.dot(self.w, self.feature_fn(state, action))

    def _update(self, episode, is_final: bool):
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        x = self.feature_fn(prev_state.state, prev_state.action)
        q_sa = np.dot(self.w, x)

        if is_final:
            target = prev_state.reward
        else:
            q_next = self.q(cur_state.state, cur_state.action)
            target = prev_state.reward + self.env.gamma * q_next

        delta = target - q_sa
        self.w += ALPHA * delta * x

    def update(self, episode: list[ReplayItem], step: int) -> None:
        self._update(episode, False)

    def finalize(self, episode: list[ReplayItem], step: int) -> None:
        self._update(episode, True)


class SarsaCNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)  # always outputs 1x1
        self.fc = nn.Linear(32, num_actions)

    def forward(self, x):
        # x: (batch, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (batch, 32)
        x = self.fc(x)  # Q-values per action
        return x


class SemiGradientSarsaCNN(ApproximateTDMethod):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = "cpu",
    ) -> None:
        super().__init__(env, load_weights)

        self.model = SarsaCNN(3, 4).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ALPHA / 100)

        self.device = device

    def get_name(self) -> str:
        return "Sarsa"

    @torch.no_grad()
    def act(
        self, state: int, step: int | None = None, mask: np.ndarray | list = []
    ) -> int:
        allowed_actions = self.get_allowed_actions(mask)
        if self._train and step and random.uniform(0, 1) < self.env.eps(step):
            return random.choice(allowed_actions)
        else:
            q_values = self.model(state[0].unsqueeze(0))
            return q_values.argmax(dim=1).item()

    def _update(self, episode, is_final: bool):
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        q_values = self.model(prev_state.state[0].unsqueeze(0))
        q_sa = q_values[0, prev_state.action]

        with torch.no_grad():
            if is_final:
                target = torch.tensor(prev_state.reward, device=self.device)
            else:
                q_next = self.model(cur_state.state[0].unsqueeze(0))[
                    0, cur_state.action
                ]
                target = prev_state.reward + self.env.gamma * q_next.detach()

        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, episode: list[ReplayItem], step: int) -> None:
        self._update(episode, False)

    def finalize(self, episode: list[ReplayItem], step: int) -> None:
        self._update(episode, True)
