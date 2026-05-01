import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from rl_book.env import ObsMode


class AgentStatus(IntEnum):
    ALIVE = 0
    STOPPING = 1
    TERMINATED = 2


@dataclass
class TorchBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    mask: torch.Tensor


class VectorizedReplayBuffer:
    def __init__(self, N: int, T: int, device):
        self.obs_buf = np.empty((T, N, 2, 7, 6), dtype=np.float32)
        self.obs_next_buf = np.empty((T, N, 2, 7, 6), dtype=np.float32)
        self.act_buf = np.empty((T, N), dtype=np.int32)
        self.rew_buf = np.empty((T, N), dtype=np.float32)
        self.done_buf = np.empty((T, N), dtype=np.bool)
        self.mask_buf = np.empty((T, N, 7), dtype=np.bool)
        self.t = 0
        self.device = device
        self.N = N
        self.T = T

    def store(self, s, a, r, s_next, done, mask):
        self.obs_buf[self.t] = s
        self.act_buf[self.t] = a
        self.rew_buf[self.t] = r
        self.obs_next_buf[self.t] = s_next
        self.done_buf[self.t] = done
        self.mask_buf[self.t] = mask

        self.t += 1

    def reset(self):
        self.t = 0
        self.act_buf.fill(-1)

    def get_batch(self, device):
        states = torch.from_numpy(self.obs_buf[: self.t]).flatten(0, 1).to(self.device)
        actions = (
            torch.from_numpy(self.act_buf[: self.t])
            .flatten(0, 1)
            .to(self.device)
            .long()
        )
        rewards = torch.from_numpy(self.rew_buf[: self.t]).flatten(0, 1).to(self.device)
        next_states = (
            torch.from_numpy(self.obs_next_buf[: self.t]).flatten(0, 1).to(self.device)
        )
        dones = torch.from_numpy(self.done_buf[: self.t]).flatten(0, 1).to(self.device)
        mask = torch.from_numpy(self.mask_buf[: self.t]).flatten(0, 1).to(self.device)

        valid_mask = actions != -1

        states = states[valid_mask]
        actions = actions[valid_mask]
        rewards = rewards[valid_mask]
        next_states = next_states[valid_mask]
        dones = dones[valid_mask]
        mask = mask[valid_mask]

        # TODO: sanity check

        return TorchBatch(states, actions, rewards, next_states, dones, mask)


class VectorizedEnv:  # TODO: Generic, ABC
    def __init__(self, env_fn, num_envs):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.status = np.full(num_envs, AgentStatus.TERMINATED, dtype=np.int32)

        self.method_idx = -1
        self.opponent_idx = -1
        self.num_envs = num_envs

        num_actions = self.envs[0].get_action_space_len()

        self.obs_np = np.empty((num_envs, *self.get_obs_shape()), dtype=np.float32)
        self.mask_np = np.empty((num_envs, num_actions), dtype=np.int8)
        self.rew_np = np.empty((num_envs,), dtype=np.float32)
        self.done_np = np.empty((num_envs,), dtype=np.bool_)

    # @abstractmethod
    def get_obs_shape(self) -> tuple[int, ...]:
        """Return per-env observation shape."""
        return (2, 7, 6)

    def reset(self, methods, zoo):
        for env in self.envs:
            env.env.reset()

        self.status.fill(AgentStatus.ALIVE)

        # Randomly sample method, player position and opponent
        self.method_idx = random.randint(0, len(methods) - 1)
        self.player_pos = random.randint(0, 1)
        self.opponent_idx = random.randint(0, len(zoo) - 1)
        self.opponent = zoo[self.opponent_idx]

    def agent_selection(self):
        agent_selections = [
            env.env.agent_selection
            for idx, env in enumerate(self.envs)
            if self.status[idx] != AgentStatus.TERMINATED
        ]
        agent_selections = [s for s in agent_selections if s is not None]

        if not agent_selections:
            return None

        first_agent_selection = agent_selections[0]
        assert all(
            s == first_agent_selection for s in agent_selections
        ), f"Inconsistent agent_selection values: {agent_selections}"

        return first_agent_selection

    def step(self, action):
        alive_idxs = np.flatnonzero(self.status != AgentStatus.TERMINATED)
        for idx in alive_idxs:
            self.envs[idx].env.step(
                action[idx] if self.status[idx] == AgentStatus.ALIVE else None
            )

    def last(self):
        self.done_np.fill(True)

        alive_idxs = np.flatnonzero(self.status != AgentStatus.TERMINATED)
        for idx in alive_idxs:
            (observation, reward, termination, truncation, _,) = self.envs[
                idx
            ].env.last()  # type: ignore

            self.obs_np[idx] = (
                self.envs[0]
                .obs_to_state(observation["observation"], obs_mode=ObsMode.RASTERIZED)
                .detach()
                .cpu()
                .numpy()
            )
            self.mask_np[idx] = observation["action_mask"].astype(
                bool, copy=False
            )  # why?
            self.rew_np[idx] = reward
            self.done_np[idx] = bool(termination or truncation)

        self.status[self.status == AgentStatus.STOPPING] = AgentStatus.TERMINATED

        return (
            self.obs_np.copy(),
            self.mask_np.copy(),
            self.rew_np.copy(),
            self.done_np.copy(),
        )

    def rewards(self, idx, is_player):
        player = self.envs[0].players[
            self.player_pos if is_player else 1 - self.player_pos
        ]
        return self.envs[idx].env.rewards[player]

    def max_game_steps(self):
        return 42

    def is_player(self, player):
        return player == self.envs[0].players[self.player_pos]

    def get_game_result(self, reward):
        return self.envs[0].get_game_result(reward)  # TODO: static
