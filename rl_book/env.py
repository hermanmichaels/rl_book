import numpy as np
from gymnasium.core import Env  # TODO: or any other env
from gymnasium.spaces import Discrete


class ParametrizedEnv:
    """Custom wrapper around Gymnasium (or other) envs."""

    def __init__(self, env: Env, gamma: float, eps_decay: bool) -> None:
        self.env = env
        self.gamma = gamma
        self.eps_end: float = 0.05
        self.eps_start: float = 0.9  # TOOD: 1 crashes with MC
        self.num_decay_steps: int = 1000
        self.eps_decay = eps_decay

    def eps(self, step: int) -> float:
        """Returns exploration factor depending on current step.

        Args:
            step: current step

        Returns:
            - constant value if no exploration decay
            - otherwise linearly decaying value
        """
        return (
            self.eps_end
            if not self.eps_decay
            else max(
                self.eps_end,
                self.eps_start
                - step * (self.eps_start - self.eps_end) / self.num_decay_steps,
            )
        )

    def obs_to_state(state):
        return state

    def get_action_space_len(self) -> int:
        return self.env.action_space.n

    def get_observation_space_len(self) -> int:
        return self.env.observation_space.n


class GridWorldEnv(ParametrizedEnv):
    """Env wrapper for "Grid world"."""

    def __init__(
        self, env: Env, gamma: float, eps_decay: bool, intermediate_rewards: bool
    ) -> None:
        super().__init__(env, gamma, eps_decay)

        self.intermediate_rewards = intermediate_rewards

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


class MultiPlayerEnv(ParametrizedEnv):
    """Wrapper around multi-player game envs."""

    def __init__(
        self, env: Env, gamma: float, players: list[str]
    ) -> None:  # TODO: wrong env
        super().__init__(env, gamma)
        self.players = players

    def get_action_space_len(self) -> int:
        return self.env.action_space(self.players[0]).n

    def get_observation_space_len(self) -> int:
        return self.env.observation_space(self.players[0]).n


class TicTacToeEnv(MultiPlayerEnv):
    """TicTacToe env."""

    def __init__(self, env: Env, gamma=0.95):
        super().__init__(env, gamma, ["player_1", "player_2"])

    def obs_to_state(self, obs, start_pos=None):
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

        if start_pos:
            state_flat.append(start_pos)

        # Convert base-3 list to integer
        state = 0
        for i, val in enumerate(state_flat):
            state += val * (3**i)
        return state


class ConnectFourEnv(MultiPlayerEnv):
    """ConnectFour env."""

    def __init__(self, env: Env, gamma=0.95) -> None:
        super().__init__(env, gamma, ["player_0", "player_1"])

    def get_observation_space_len(self):
        return 3 ** (6 * 7 * 3 * 2)  # TODO?

    def obs_to_state(self, state, start_pos=None):
        board = state  # shape: (6, 7, 2)
        state_flat = []

        for row in range(6):
            for col in range(7):
                if board[row][col][0] == 1:
                    state_flat.append(1)  # player 1
                elif board[row][col][1] == 1:
                    state_flat.append(2)  # player 2
                else:
                    state_flat.append(0)  # empty

        if start_pos is not None:
            state_flat.append(start_pos)

        # Convert to base-3 integer
        state = 0
        for i, val in enumerate(state_flat):
            state += val * (3**i)

        return state


class MethodStats:
    def __init__(self, method):
        self.method = method
        self.wins = 0
        self.picks = 0

    def update_pick(self):
        self.picks += 1

    def update_win(
        self,
    ):
        self.wins += 1

    def get_win_ratio(self):
        return self.wins / (self.picks + 1)

    def clone(self):
        cloned = MethodStats(self.method.clone())
        cloned.wins = self.wins
        cloned.picks = self.picks
        return cloned
