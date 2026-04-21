from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Generic, Literal, TypeVar, cast, overload

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import Env
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.spaces import Box, Discrete


# Toggle different observation "modes", such as
# "default" (int values) and rasterized images.
class ObsMode(Enum):
    INVALID = 0
    DEFAULT = 1
    RASTERIZED = 2


S = TypeVar("S")
GAMMA = 0.97


@overload
def generate_random_grid_world_env(
    n: int,
    extra_rewards: bool,
    eps_decay: bool,
    obs_mode: Literal[ObsMode.DEFAULT],
    device: torch.device = torch.device("cpu"),
) -> tuple["GridWorldEnv[int]", list[str]]:
    ...


@overload
def generate_random_grid_world_env(
    n: int,
    extra_rewards: bool,
    eps_decay: bool,
    obs_mode: Literal[ObsMode.RASTERIZED],
    device: torch.device = torch.device("cpu"),
) -> tuple["GridWorldEnv[tuple[torch.Tensor, int]]", list[str]]:
    ...


@overload
def generate_random_grid_world_env(
    n: int,
    extra_rewards: bool,
    eps_decay: bool,
    obs_mode: ObsMode,
    device: torch.device = torch.device("cpu"),
) -> tuple["GridWorldEnv[int] | GridWorldEnv[tuple[torch.Tensor, int]]", list[str],]:
    ...


def generate_random_grid_world_env(
    n: int,
    extra_rewards: bool,
    eps_decay: bool,
    obs_mode: ObsMode,
    device: torch.device = torch.device("cpu"),
) -> tuple["GridWorldEnv", list[str]]:
    desc = generate_random_map(size=n)
    gym_env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
    )
    return (
        GridWorldEnv(
            gym_env,
            GAMMA,
            intermediate_rewards=extra_rewards,
            eps_decay=eps_decay,
            obs_mode=obs_mode,
            device=device,
        ),
        desc,
    )


class ParametrizedEnv(Generic[S]):
    """Custom wrapper around Gymnasium (or other) envs."""

    def __init__(self, env: Env, gamma: float, eps_decay: bool) -> None:
        self.env = env
        self.gamma = gamma
        self.eps_end: float = 0.05
        self.eps_start: float = 0.9  # TOOD: 1 crashes with MC
        self.num_decay_steps: int = 10000
        self.eps_decay = eps_decay

    def eps(self, step: int) -> float:
        """Returns exploration factor depending on current step.

        Args:
            step: current step

        Returns:
            - constant value if no exploration decay
            - otherwise linearly decaying value
        """
        # step = 100000
        return (
            self.eps_end
            if not self.eps_decay
            else max(
                self.eps_end,
                self.eps_start
                - step * (self.eps_start - self.eps_end) / self.num_decay_steps,
            )
        )

    def step(self, action: int, old_obs: S) -> tuple[S, float, bool, bool, dict]:
        raise NotImplementedError

    def get_action_space_len(self) -> int:
        raise NotImplementedError

    def get_observation_space_len(self) -> int:
        raise NotImplementedError

    def get_max_num_steps(self) -> int:
        raise NotImplementedError


class GridWorldEnv(ParametrizedEnv[S], Generic[S]):
    """Env wrapper for "Grid world"."""

    def __init__(
        self,
        env: Env,
        gamma: float,
        eps_decay: bool,
        intermediate_rewards: bool,
        obs_mode: ObsMode,
        device: torch.device,
    ) -> None:
        if obs_mode == ObsMode.RASTERIZED:
            super().__init__(GridWorldImageWrapper(env, device), gamma, eps_decay)
        else:
            super().__init__(env, gamma, eps_decay)

        self.intermediate_rewards = intermediate_rewards
        self.grid_size = env.unwrapped.desc.shape[0]  # type: ignore[attr-defined]
        self.obs_mode = obs_mode

    def normalized_grid_position_sum(self, observation: int) -> float:
        """Computes the normalized row / column index of the passed observation.
        Used for reward heuristics under the assumption that a higher such
        value is better / closer to the goal.
        """
        return (
            observation // self.grid_size + observation % self.grid_size
        ) / self.grid_size

    def step(self, action: int, old_obs: S) -> tuple[S, float, bool, bool, dict]:
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
            if self.obs_mode == ObsMode.RASTERIZED:
                assert isinstance(old_obs, tuple)
                _, obs_for_intermediate = observation
                _, old_obs_for_intermediate = old_obs
            else:
                assert isinstance(old_obs, int)
                obs_for_intermediate = observation
                old_obs_for_intermediate = old_obs

            reward += self.normalized_grid_position_sum(
                obs_for_intermediate
            ) - self.normalized_grid_position_sum(old_obs_for_intermediate)
        return observation, reward, terminated, truncated, info

    def get_action_space_len(self) -> int:
        assert isinstance(self.env.action_space, Discrete)
        return cast(int, self.env.action_space.n)

    def get_observation_space_len(self) -> int:
        assert isinstance(self.env.observation_space, Discrete)
        return cast(int, self.env.observation_space.n)

    def get_max_num_steps(self) -> int:
        return self.grid_size**2 * 4


class GridWorldImageWrapper(gym.ObservationWrapper):
    """Wrapper around GridWorldEnv to provide rasterized images as observations."""

    def __init__(self, env: Env, device: torch.device):
        super().__init__(env)

        self.H = env.unwrapped.desc.shape[0]  # type: ignore[attr-defined]
        self.W = env.unwrapped.desc.shape[1]  # type: ignore[attr-defined]
        self.device = device

        # Precompute walls and goal
        walls = set()
        goal_pos = None
        for r in range(self.H):
            for c in range(self.W):
                if env.unwrapped.desc[r, c] == b"H":  # type: ignore[attr-defined]
                    walls.add((r, c))
                elif env.unwrapped.desc[r, c] == b"G":  # type: ignore[attr-defined]
                    goal_pos = (r, c)
        assert goal_pos is not None, "No goal found, this should not happen"

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(3, self.H, self.W),
            dtype=np.float32,
        )

        # Create a base obs with goal and walls precomputed,
        # since this information is static
        self.base_obs = torch.zeros((3, self.H, self.W), device=device)
        self.base_obs[1, goal_pos[0], goal_pos[1]] = 1.0
        for (r, c) in walls:
            self.base_obs[2, r, c] = 1.0

    def observation(self, state: int) -> tuple[torch.Tensor, int]:
        """Computes the rasterized image observation.

        Args:
            state: original env state

        Returns:
            - rasterized state
            - as well as original int state (for intermediate rewards)
        """
        # Compute agent position from integer state
        x = state // self.W
        y = state % self.W

        obs = self.base_obs.clone()
        obs[0, x, y] = 1.0

        return (
            obs,
            state,
        )


class GameResult(Enum):
    INVALID = 0
    WIN = 1
    DRAW = 2
    LOSS = 3


class MultiPlayerEnv(ParametrizedEnv[S], Generic[S]):
    """Wrapper around multi-player game envs.
    Atm only 2-player games are supported."""

    def __init__(
        self, env: Env, gamma: float, players: list[str], device=torch.device("cpu")
    ) -> None:
        super().__init__(env, gamma, True)
        if not len(players) == 2:
            raise ValueError(f"Expected two players, but got {players}")
        self.players = players
        self.device = device

    def get_action_space_len(self) -> int:
        return self.env.action_space(self.players[0]).n  # type: ignore

    def get_observation_space_len(self) -> int:
        raise NotImplementedError

    def obs_to_state(
        self, obs: Any, player_pos: int = 0, obs_mode: ObsMode = ObsMode.DEFAULT
    ) -> S:
        raise NotImplementedError

    def get_game_result(self, reward) -> GameResult:
        raise NotImplementedError

    def user_query(self):
        # TODO: potentially mask non-avail actions
        raise NotImplementedError



class TicTacToeEnv(MultiPlayerEnv[int | torch.Tensor]):
    """TicTacToe env."""

    def __init__(self, env: Env, gamma=0.95, device=torch.device("cpu")):
        super().__init__(env, gamma, ["player_1", "player_2"], device)

    def obs_to_state(
        self, obs: Any, start_pos: int = 0, obs_mode: ObsMode = ObsMode.DEFAULT
    ) -> int | torch.Tensor:
        if obs_mode == ObsMode.DEFAULT:
            board = obs  # shape: (3, 3, 2)
            state_flat = []

            for row in range(3):
                for col in range(3):
                    if board[row][col][0] == 1:
                        state_flat.append(1)  # player 1
                    elif board[row][col][1] == 1:
                        state_flat.append(2)  # player 2
                    else:
                        state_flat.append(0)  # empty

            # Convert base-3 list to integer
            state = 0
            for i, val in enumerate(state_flat):
                state += val * (3**i)

            return state
        elif obs_mode == ObsMode.RASTERIZED:
            return torch.as_tensor(
                np.transpose(obs, [2, 1, 0]), device=self.device
            ).float()
        else:
            raise ValueError(f"Got unexpected obs_mode {obs_mode}")

    def get_game_result(self, reward: float) -> GameResult:
        if reward == 1:
            return GameResult.WIN
        elif reward == 0:
            return GameResult.DRAW
        elif reward == -1:
            return GameResult.LOSS
        else:
            raise ValueError(f"Unexpected game-ending reward {reward}")

    def user_query(self) -> str:
        return "Please indicate in which cell to place your symbol.\n\
        The cells are indexed as follows:\n\
        0 | 3 | 6\n\
        _________\n\
        1 | 4 | 7\n\
        _________\n\
        2 | 5 | 8"



class ConnectFourEnv(MultiPlayerEnv[int | torch.Tensor]):
    """ConnectFour env."""

    def __init__(self, env: Env, gamma=0.95, device=torch.device("cpu")) -> None:
        super().__init__(env, gamma, ["player_0", "player_1"], device)
        self.c = 0

    def obs_to_state(
        self, obs: Any, start_pos: int = 0, obs_mode: ObsMode = ObsMode.DEFAULT
    ) -> int | torch.Tensor:

        if obs_mode == ObsMode.DEFAULT:
            board = obs  # shape: (6, 7, 2)

            state_flat = []

            for row in range(6):
                for col in range(7):
                    if board[row][col][0] == 1:
                        state_flat.append(1)  # player 1
                    elif board[row][col][1] == 1:
                        state_flat.append(2)  # player 2
                    else:
                        state_flat.append(0)  # empty

            state_flat.append(start_pos)  # TODO: remove!

            # Convert to base-3 integer
            state_encoded = 0
            for i, val in enumerate(state_flat):
                state_encoded += val * (3**i)

            return state_encoded
        elif obs_mode == ObsMode.RASTERIZED:
            # print("###")
            # save_connect4_obs(obs, f"plots/{self.c}.png")
            # assert False
            # TODO: not to tensor
            self.c += 1
            if obs.ndim == 3:
                res = torch.as_tensor(
                    np.transpose(obs, [2, 1, 0]), device=self.device
                ).float()
            else:
                res = torch.as_tensor(
                    np.transpose(obs, [0, 3, 2, 1]), device=self.device
                ).float()
            return res
        raise ValueError(f"Got unexpected obs_mode {obs_mode}")

    def get_game_result(self, reward: float) -> GameResult:
        if reward == 1:
            return GameResult.WIN
        elif reward == 0:
            return GameResult.DRAW
        elif reward == -1:
            return GameResult.LOSS
        else:
            raise ValueError(f"Unexpected game ending reward {reward}")

    def user_query(self) -> str:
        return (
            "Please indicate in which column in which to drop the next token (0 - 6):"
        )



