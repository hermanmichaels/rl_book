import copy
import pickle
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Type, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

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
        **kwargs: object,
    ) -> None:
        self.device = device

        super().__init__(env, load_weights, **kwargs)

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
            probs = allowed_actions.float()

            row_sum = probs.sum(dim=1)
            invalid_zero_sum = row_sum <= 0
            if invalid_zero_sum.any():
                probs[invalid_zero_sum] += 1 / probs.shape[1]

            return torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            q_values = self.q(state, mask)

            # Sample uniformly in case of ties
            max_q = q_values.max(dim=1, keepdim=True).values
            probs = (max_q == q_values).float()  # [N, A]
            row_sum = probs.sum(dim=1, keepdim=True)

            probs /= row_sum.clamp_min(0)

            return torch.multinomial(probs, 1)[:, 0]

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
        **kwargs: object,
    ) -> None:
        self.num_states = env.get_observation_space_len()
        self.num_actions = env.get_action_space_len()
        self.w = np.zeros(self.num_states * self.num_actions)

        super().__init__(env, load_weights, device, **kwargs)

    @override
    def clone(self) -> "SemiGradientSarsaLinear":
        cloned = self.__class__(self.env, False, self.device)
        cloned.w = np.copy(self.w)
        return cloned

    @override
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

    @override
    def q(self, state: S, allowed_actions: np.ndarray) -> torch.Tensor:
        q_values = torch.Tensor(
            [np.dot(self.w, self.feature_fn(state, a)) for a in allowed_actions]
        )
        return q_values

    @override
    def update(self, episode: list[ReplayItem[S]], step: int) -> None:
        if len(episode) <= 2:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        x = self.feature_fn(prev_state.state, prev_state.action)
        q_sa = np.dot(self.w, x)

        all_actions = np.asarray([a for a in range(self.num_actions)])
        q_next = self.q(cur_state.state, all_actions)[cur_state.action].item()
        target = prev_state.reward + self.env.gamma * q_next

        delta = target - q_sa
        self.w += ALPHA * delta * x

    @override
    def finalize(self, episode: list[ReplayItem[S]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        x = self.feature_fn(cur_state.state, cur_state.action)
        q_sa = np.dot(self.w, x)

        target = cur_state.reward

        delta = target - q_sa
        self.w += ALPHA * delta * x

    def _get_save_data(self) -> Any:
        return self.w

    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            self.w = pickle.load(f)


class SemiGradientSarsaCNN(ApproximateTDMethod[S], Generic[S, T]):
    obs_mode: ClassVar[ObsMode] = ObsMode.RASTERIZED

    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        network_class: Type[T] | None = None,
        **kwargs: object,
    ) -> None:
        assert network_class is not None, "network_class must be set"

        num_actions = env.get_action_space_len()
        self.model = network_class(num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.network_class = network_class
        self.all_actions = np.ones((3, 7))  # TODO N!!

        super().__init__(env, load_weights, device, **kwargs)

    @override
    def clone(self) -> "SemiGradientSarsaCNN":
        cloned = self.__class__(
            self.env,
            False,
            self.device,
            self.network_class,
        )
        cloned.model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        return cloned

    @override
    def get_name(self) -> str:
        return "SemiGradientSarsa-CNN"

    @override
    def q(
        self,
        state: S,
        mask: np.ndarray,
    ) -> torch.Tensor:
        assert mask.ndim == 2
        if isinstance(state, tuple):
            q_values = self.model(state[0].unsqueeze(0))
        elif isinstance(state, torch.Tensor):
            if len(state.shape) < 4:
                # Insert batch dimension if not present
                state = state.unsqueeze(0)
            q_values = self.model(state)
        else:
            raise ValueError(f"Got unexpected type {type(state)}")
        q_masked = q_values.masked_fill(
            ~torch.Tensor(mask).bool().cuda(), float("-inf")
        )
        return q_masked

    @override
    def update(self, episode: list[ReplayItem[S]], step: int) -> None:
        """Executes one update step.

        Args:
            episode: current episode up to now
            is_final: true when episode has ended
        """
        if len(episode) <= 2:
            return

        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        q_values = self.q(prev_state.state, self.all_actions)
        q_sa = q_values[0, prev_state.action]

        with torch.no_grad():
            q_next = self.q(cur_state.state, self.all_actions)[0, cur_state.action]
            target = prev_state.reward + self.env.gamma * q_next.detach()

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

    def finalize(self, episode: list[ReplayItem[S]], step: int) -> None:
        if len(episode) <= 1:
            return

        cur_state = episode[len(episode) - 1]

        q_values = self.q(cur_state.state, self.all_actions)
        q_sa = q_values[0, cur_state.action]

        with torch.no_grad():
            target = torch.tensor(cur_state.reward, device=self.device)

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_save_data(self) -> Any:
        return self.model.state_dict()

    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            state_dict = pickle.load(f)
            self.model.load_state_dict(state_dict)

    def batch_update(self, batch):
        repeat_len = batch.states.shape[0]

        q = self.q(batch.states, np.repeat(self.all_actions[:1], repeat_len, axis=0))
        # Predicted Q values of selected actions
        q_sa = q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # [mbs, num_actions]
            legal = batch.mask.bool()
            # [mbs]
            has_legal = legal.any(dim=1)

            q_next = self.q(
                batch.next_states, np.repeat(self.all_actions[:1], repeat_len, axis=0)
            )
            q_next_masked = q_next.masked_fill(~legal, float("-inf"))

            # [mbs]
            max_next = q_next_masked.max(dim=1).values
            max_next = torch.where(has_legal, max_next, torch.zeros_like(max_next))

            target = batch.rewards + self.env.gamma * (~batch.dones).float() * max_next

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()


class SemiGradientSarsaNCNN(ApproximateTDMethod[S], Generic[S, T]):
    obs_mode: ClassVar[ObsMode] = ObsMode.RASTERIZED

    def __init__(
        self,
        env: ParametrizedEnv,
        load_weights: bool = False,
        device: torch.device = torch.device("cpu"),
        network_class: Type[T] | None = None,
        n: int = 3,
        **kwargs: object,
    ) -> None:
        assert network_class is not None, "network_class must be set"

        self.num_actions = env.get_action_space_len()
        self.model = network_class(self.num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ALPHA / 10)
        self.network_class = network_class
        self.n = n

        super().__init__(env, load_weights, device, **kwargs)

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
        if tau is None:
            # tau is set when finalizing the episode - otherwise pick
            # the correct update step here.
            tau = len(episode) - self.n - 1

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

                if tau + self.n < len(episode):
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
        for tau in range(len(episode) - self.n, len(episode)):
            self._update(episode, tau)

    def _get_save_data(self) -> Any:
        return self.model.state_dict()

    def _load_weights(self, save_path: str) -> None:
        with open(save_path, "rb") as f:
            state_dict = pickle.load(f)
            self.model.load_state_dict(state_dict)
