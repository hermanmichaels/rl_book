import copy
import pickle
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_book.env import ObsMode, ParametrizedEnv
from rl_book.methods.method import RLMethod
from rl_book.replay_utils import ReplayItem

ALPHA = 0.1

S = TypeVar("S")
T = TypeVar("T", bound=nn.Module)


class ApproximateTDMethod(RLMethod[S], Generic[S], ABC):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(env, load_weights, device)

    def clone(self) -> "ApproximateTDMethod":
        cloned = self.__class__(self.env, False)
        return cloned

    @torch.no_grad()
    def act(
        self,
        state: S,
        step: int | None = None,
        mask: np.ndarray | list = [],
    ) -> int:
        allowed_actions = self.get_allowed_actions(mask)
        if self._train and step and random.uniform(0, 1) < self.env.eps(step):
            return random.choice(allowed_actions)
        else:
            q_values = self.q(state, allowed_actions)
            # Sample uniformly in case of ties
            max_q = q_values.max()
            probs = (q_values == max_q).float()
            probs /= probs.sum()
            chosen_idx: int = int(torch.multinomial(probs, 1).item())
            all_actions = self.get_allowed_actions([])
            return all_actions[chosen_idx]

    def _get_save_data(self) -> Any:
        pass

    def _load_weights(self, save_path: str) -> None:
        pass

    @abstractmethod
    def q(self, state: S, allowed_actions: np.ndarray) -> torch.Tensor:
        """Computes q function.

        Args:
            state: current state
            allowed_actions: allowed actions

        Returns:
            tensor containing q values of all allowed actions
        """


class SemiGradientSarsaLinear(ApproximateTDMethod[S], Generic[S]):
    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(env, load_weights, device)

        self.num_states = env.get_observation_space_len()
        self.num_actions = env.get_action_space_len()
        self.w = np.zeros(self.num_states * self.num_actions)

    def clone(self) -> "SemiGradientSarsaLinear":
        cloned = self.__class__(self.env, False)
        cloned.w = np.copy(self.w)
        return cloned

    def get_name(self) -> str:
        return "SemiGradientSarsa-Linear"

    def feature_fn(self, state: S, action: int) -> np.ndarray:
        """Simple feature function returning a one-hot representation
        of state and action.

        Args:
            state: current state
            action: action to take

        Returns:
            feature vector
        """
        assert isinstance(state, int)
        x = np.zeros(self.num_states * self.num_actions)
        idx = state * self.num_actions + action
        x[idx] = 1.0
        return x

    def q(self, state: S, allowed_actions: np.ndarray) -> torch.Tensor:
        # TODO: allowed?
        q_values = torch.Tensor(
            [np.dot(self.w, self.feature_fn(state, a)) for a in allowed_actions]
        )
        return q_values

    def _update(self, episode: list[ReplayItem[S]], is_final: bool) -> None:
        """Executes one update step.

        Args:
            episode: current episode up to now
            is_final: true when episode has ended
        """
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        x = self.feature_fn(prev_state.state, prev_state.action)
        q_sa = np.dot(self.w, x)

        if is_final:
            target = prev_state.reward
        else:
            all_actions = np.asarray([a for a in range(self.num_actions)])
            q_next = self.q(cur_state.state, all_actions)[cur_state.action].item()
            target = prev_state.reward + self.env.gamma * q_next

        delta = target - q_sa
        self.w += ALPHA * delta * x

    def update(self, episode: list[ReplayItem[S]], step: int) -> None:
        self._update(episode, False)

    def finalize(self, episode: list[ReplayItem[S]], step: int) -> None:
        self._update(episode, True)

    def _get_save_data(self) -> Any:
        return self.w

    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            self.w = pickle.load(f)


class GridWorldCNN(nn.Module):
    """Simple CNN to process rasterized GridWorld images and output Q values."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_actions)

    def forward(self, x: torch.Tensor):
        """Forward call.

        Args:
            x: input tensor [bs, C, H, W]

        Returns:
            Q values [bs, num_actions]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CNNTicTacToe(nn.Module):
    """Simple CNN to process rasterized GridWorld images and output Q values."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor):
        """Forward call.

        Args:
            x: input tensor [bs, C, H, W]

        Returns:
            Q values [bs, num_actions]
        """
        return self.net(x)




class CNNConnectFour(nn.Module):
    """
    Lightweight CNN for Connect4 Q-values.
    Input: [bs, 2, 6, 7] (player-to-move pieces, opponent pieces)
    Output: [bs, 7] Q-values for columns.
    """

    def __init__(self, num_actions: int = 7, hidden: int = 64, pooled: int = 2) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

        # Keep coarse spatial layout (2x2 works well for 6x7)
        self.pool = nn.AdaptiveAvgPool2d((pooled, pooled))
        self.fc = nn.Linear(hidden * pooled * pooled, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

class SemiGradientSarsaCNN(ApproximateTDMethod[S], Generic[S]):
    obs_mode: ClassVar[ObsMode] = ObsMode.RASTERIZED

    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        network_class: Type[T] | None = None,
    ) -> None:
        # print("INIT")
        self.model = network_class(9).to(device) # TODO
        
        super().__init__(env, load_weights, device)
        # print("INIT2")

        # self._avg()

        num_actions = self.env.get_action_space_len()
        assert network_class is not None, "network_class must be set"

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ALPHA / 10)
        self.network_class = network_class
        self.all_actions = np.asarray([a for a in range(num_actions)])

    def _avg(self):
        total_sum = 0.0
        total_count = 0

        with torch.no_grad():
            for param in self.model.parameters():
                total_sum += param.sum().item()
                total_count += param.numel()

        avg_weight = total_sum / total_count
        print("Average model weight:", avg_weight)

    def clone(self) -> "SemiGradientSarsaCNN":
        cloned = self.__class__(
            self.env,
            False,
            self.device,
            self.network_class,
        )
        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        return cloned

    def get_name(self) -> str:
        return "SemiGradientSarsa-CNN"

    def q(
        self,
        state: S,
        allowed_actions: np.ndarray,  # todo: wrong
    ) -> torch.Tensor:
        if isinstance(state, tuple):
            q_values = self.model(state[0].unsqueeze(0))
        elif isinstance(state, torch.Tensor):
            q_values = self.model(state.unsqueeze(0))
        else:
            raise ValueError(f"Got unexpected type {type(state)}")
        mask = torch.zeros_like(q_values).bool()
        mask[:, allowed_actions] = 1.0
        q_masked = q_values.masked_fill(~mask, float("-inf"))
        return q_masked

    def _update(self, episode: list[ReplayItem[S]], is_final: bool) -> None:
        """Executes one update step.

        Args:
            episode: current episode up to now
            is_final: true when episode has ended
        """
        if len(episode) <= 1:
            return

        prev_state = episode[len(episode) - 2] if not is_final else episode[len(episode) - 1]
        cur_state = episode[len(episode) - 1]

        q_values = self.q(prev_state.state, self.all_actions)
        q_sa = q_values[0, prev_state.action]

        with torch.no_grad():
            if is_final:
                target = torch.tensor(prev_state.reward, device=self.device)
            else:
                q_next = self.q(cur_state.state, self.all_actions)[0, cur_state.action]
                target = prev_state.reward + self.env.gamma * q_next.detach()

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

    def update(self, episode: list[ReplayItem[S]], step: int) -> None:
        self._update(episode, False)

    def finalize(self, episode: list[ReplayItem[S]], step: int) -> None:
        self._update(episode, True)

    def _get_save_data(self) -> Any:
        return self.model.state_dict()

    def _load_weights(self, save_path: str) -> None:
        # import ipdb
        # ipdb.set_trace()
        with open(save_path, "rb") as f:
            state_dict = pickle.load(f)
            print(state_dict)
            self.model.load_state_dict(state_dict)


class SemiGradientSarsaNCNN(ApproximateTDMethod[S], Generic[S]):
    obs_mode: ClassVar[ObsMode] = ObsMode.RASTERIZED

    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        network_class: Type[T] | None = None,
        n: int = 3,
    ) -> None:
        super().__init__(env, load_weights, device)

        self.num_actions = self.env.get_action_space_len()
        assert network_class is not None, "network_class must be set"
        self.model = network_class(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ALPHA / 10)
        self.network_class = network_class
        self.n = n

    def clone(self) -> "SemiGradientSarsaNCNN":
        cloned = self.__class__(
            self.env, False, self.device, self.network_class, self.n
        )
        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        return cloned

    def get_name(self) -> str:
        return "SemiGradientSarsaN-CNN"

    def q(
        self,
        state: S,
        allowed_actions: np.ndarray,  # todo: wrong
    ) -> torch.Tensor:
        if isinstance(state, tuple):
            q_values = self.model(state[0].unsqueeze(0))
        elif isinstance(state, torch.Tensor):
            q_values = self.model(state.unsqueeze(0))
        else:
            raise ValueError(f"Got unexpected type {type(state)}")
        mask = torch.zeros_like(q_values).bool()
        mask[:, allowed_actions] = 1.0
        q_masked = q_values.masked_fill(~mask, float("-inf"))
        return q_masked

    def _update(
        self,
        episode: list[ReplayItem[S]],
        tau: int | None = None,
    ) -> None:
        """Executes one update step.

        Args:
            episode: current episode up to now
            tau: current timestep to update
        """
        is_final = True
        if tau is None:
            # tau is set when finalizing the episode - otherwise pick
            # the correct update step here.
            tau = len(episode) - self.n - 1
            is_final = False

        if tau >= 0:
            all_actions = np.asarray([a for a in range(self.num_actions)])
            with torch.no_grad():
                G = sum(
                    [
                        episode[i].reward * self.env.gamma ** (i - tau)
                        for i in range(tau, min(tau + self.n, len(episode)))
                    ]
                )
                G_torch = torch.tensor(G, dtype=torch.float32, device=self.device)
                if not is_final:
                    G_torch = (
                        G_torch
                        + self.env.gamma**self.n
                        * self.q(episode[tau + self.n].state, all_actions)[
                            0, episode[tau + self.n].action
                        ].detach()
                    )
                target = G_torch

            q_values = self.q(episode[tau].state, all_actions)
            q_sa = q_values[0, episode[tau].action]

            loss = F.mse_loss(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update(self, episode: list[ReplayItem[S]], step: int) -> None:
        self._update(episode)

    def finalize(self, episode: list[ReplayItem[S]], step: int) -> None:
        # Replay has terminated - still finish updating the values
        # by going over the remaining episode.
        for tau in range(len(episode) - self.n - 1, len(episode)):
            self._update(episode, tau)

    def _get_save_data(self) -> Any:
        return self.model.state_dict()

    def _load_weights(self, save_path: str) -> None:
        state_dict = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(state_dict)


# RUNS:
# - set ticks!
# 5, 26: new methods
# 10, 50: best methods: prob. previous best methods + SarsaCNN
# Then show how much SarsaCNN can be scaled, e.g. up to 100, ... - maybe first another run with other best-performing methods
# Maybe change receptive field
