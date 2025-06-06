import random
from typing import Callable

import numpy as np

from rl_book.env import ParametrizedEnv
from rl_book.gym_utils import get_observation_action_space
from rl_book.methods.method import Algorithm
from rl_book.methods.method_wrapper import with_default_values
from rl_book.utils import get_eps_greedy_action, get_policy

from collections import defaultdict
import copy

ALPHA = 0.1

class TDMethod(Algorithm):
    def clone(self):
        cloned = self.__class__()
        cloned.Q = copy.deepcopy(self.Q)
        return cloned

class Sarsa(TDMethod):
    def __init__(self):
        self.Q = defaultdict(float)

    def update(self, episode):
        if len(episode) <= 1:
            return
        
        prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 1]

        if np.sum(cur_state.mask) > 0:
            action_new = self.act(cur_state.state, cur_state.mask)
            q_next = self.Q[cur_state.state, action_new]
        else:
            q_next = 0
        # import ipdb
        # ipdb.set_trace()
            

        # if prev_state.reward > 0:
        #     import ipdb
        #     ipdb.set_trace()

        self.Q[prev_state.state, prev_state.action] = self.Q[prev_state.state, prev_state.action] + ALPHA * (
            float(prev_state.reward) + 0.95 * q_next - self.Q[prev_state.state, prev_state.action]
        )

    def finalize(self, episode):
        self.update(episode)

    # TODO: share
    def act(self, state, mask):
        q_values = [self.Q[state, a] for a in np.nonzero(mask)[0].tolist()]
        # import ipdb
        # ipdb.set_trace()
        max_q = max(q_values)

        # import ipdb
        # ipdb.set_trace()
        eps_greedy = random.randint(0, 100)

        # TODO: eps greedy?
        max_actions = [a for a, q in zip(np.nonzero(mask)[0].tolist(), q_values) if q == max_q or eps_greedy <= 5]

        return random.choice(max_actions)
    
    

class MinMax:
    def __init__(self, depth=1):
        self.depth = depth

    def act(self, encoded_state, mask):
        board = self.decode_state(encoded_state)
        best_score = -np.inf
        best_action = None

        for action in range(7):
            if mask[action] == 0:
                continue
            board_copy = board.copy()
            self.drop_piece(board_copy, action, 2)  # Agent plays as player 2
            score = self.minimax(board_copy, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else np.random.choice(np.nonzero(mask)[0])
    
    def update(self, episode):
        pass

    def finalize(self, episode):
        pass

    def clone(self):
        return self

    def minimax(self, board, depth, maximizing):
        winner = self.check_winner(board)
        if winner == 2:
            return 1000
        elif winner == 1:
            return -1000
        elif self.is_draw(board):
            return 0

        if depth == 0:
            return self.evaluate(board)

        valid_actions = [c for c in range(7) if board[0][c] == 0]
        if maximizing:
            max_eval = -np.inf
            for a in valid_actions:
                b_copy = board.copy()
                self.drop_piece(b_copy, a, 2)
                score = self.minimax(b_copy, depth - 1, False)
                max_eval = max(max_eval, score)
            return max_eval
        else:
            min_eval = np.inf
            for a in valid_actions:
                b_copy = board.copy()
                self.drop_piece(b_copy, a, 1)
                score = self.minimax(b_copy, depth - 1, True)
                min_eval = min(min_eval, score)
            return min_eval

    def evaluate(self, board):
        score = 0
        score += self.count_n_in_a_row(board, 2, 2) * 10
        score += self.count_n_in_a_row(board, 2, 3) * 50
        score -= self.count_n_in_a_row(board, 1, 2) * 8
        score -= self.count_n_in_a_row(board, 1, 3) * 40
        return score

    def count_n_in_a_row(self, board, player, n):
        count = 0
        for row in range(6):
            for col in range(7):
                if self.check_line(board, row, col, 0, 1, player, n): count += 1
                if self.check_line(board, row, col, 1, 0, player, n): count += 1
                if self.check_line(board, row, col, 1, 1, player, n): count += 1
                if self.check_line(board, row, col, 1, -1, player, n): count += 1
        return count

    def check_line(self, board, row, col, dr, dc, player, n):
        for i in range(n):
            r, c = row + i * dr, col + i * dc
            if not (0 <= r < 6 and 0 <= c < 7) or board[r][c] != player:
                return False
        return True

    def drop_piece(self, board, col, player):
        for row in reversed(range(6)):
            if board[row][col] == 0:
                board[row][col] = player
                return

    def check_winner(self, board):
        for row in range(6):
            for col in range(7):
                for dr, dc in [(0,1), (1,0), (1,1), (1,-1)]:
                    if all(
                        0 <= row + i * dr < 6 and
                        0 <= col + i * dc < 7 and
                        board[row + i * dr][col + i * dc] == 1
                        for i in range(4)
                    ):
                        return 1
                    if all(
                        0 <= row + i * dr < 6 and
                        0 <= col + i * dc < 7 and
                        board[row + i * dr][col + i * dc] == 2
                        for i in range(4)
                    ):
                        return 2
        return None

    def is_draw(self, board):
        return np.all(board[0] != 0)

    def decode_state(self, encoded_state):
        """Reverses obs_to_state for Connect Four"""
        state_flat = []
        for _ in range(42):
            encoded_state, rem = divmod(encoded_state, 3)
            state_flat.append(rem)

        board = np.zeros((6, 7), dtype=int)
        for idx, val in enumerate(state_flat):
            row = idx // 7
            col = idx % 7
            board[row][col] = val
        return board
    
class QLearning(TDMethod):
    def __init__(self):
        self.Q = defaultdict(float)

    def update(self, episode):
        if len(episode) <= 1:
            return
        
        # prev_state = episode[len(episode) - 2]
        cur_state = episode[len(episode) - 2]
        next_state = episode[len(episode) - 1]
        next_q = max([self.Q[next_state.state, a_] for a_ in range(len(cur_state.mask))], default=0) # TODO: maks # tood: right mask index?
        print([self.Q[next_state.state, a_] for a_ in cur_state.mask])
        if cur_state.reward > 0:
            print("##")
        # if next_q > 0 or random.randint(0, 1000) < 0 or next_state.state == 14:
        #     import ipdb
        #     ipdb.set_trace()
        self.Q[cur_state.state, cur_state.action] = self.Q[cur_state.state, cur_state.action] + ALPHA * (
                        cur_state.reward + 0.95 * next_q - self.Q[cur_state.state, cur_state.action]
                        )

    def finalize(self, episode):
        print("####")
        self.update(episode)

    def act(self, state, mask):
        q_values = [self.Q[(state, a)] for a in np.nonzero(mask)[0].tolist()]
        max_q = max(q_values)

        eps_greedy = random.randint(0, 100)

        # TODO: eps greedy?
        max_actions = [a for a, q in zip(np.nonzero(mask)[0].tolist(), q_values) if q == max_q or eps_greedy <= 5]

        return random.choice(max_actions)
    


class Random(TDMethod):
    def __init__(self):
        self.Q = defaultdict(float)

    def update(self, episode):
        pass

    def act(self, state, mask):
        q_values = [self.Q[(state, a)] for a in np.nonzero(mask)[0].tolist()]

        # TODO: eps greedy?
        max_actions = [a for a, q in zip(np.nonzero(mask)[0].tolist(), q_values)]

        return random.choice(max_actions)

@with_default_values
def sarsa(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        eps = env.eps(step)

        action = get_eps_greedy_action(Q[observation], eps)

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )
            action_new = get_eps_greedy_action(Q[observation_new], eps)
            q_next = Q[observation_new, action_new] if not terminated else 0
            Q[observation, action] = Q[observation, action] + ALPHA * (
                float(reward) + env.gamma * q_next - Q[observation, action]
            )
            observation = observation_new
            action = action_new

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step


@with_default_values
def q(
    env, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False

        cur_episode_len = 0

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q[observation], env.eps(step))
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            Q[observation, action] = Q[observation, action] + ALPHA * (
                reward + env.gamma * np.max(Q[observation_new]) - Q[observation, action]
            )
            observation = observation_new

            cur_episode_len += 1
            if cur_episode_len > 400:
                break

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q, observation_space), step


@with_default_values
def expected_sarsa(
    env: ParametrizedEnv, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q = np.zeros((observation_space.n, action_space.n))

    def _get_action_prob(Q: np.ndarray) -> float:
        return (
            Q[observation_new, a] / sum(Q[observation_new, :])
            if sum(Q[observation_new, :])
            else 1
        )

    for step in range(max_steps):
        observation, _ = env.env.reset()
        terminated = truncated = False
        action = get_eps_greedy_action(Q[observation])

        cur_episode_len = 0

        while not terminated and not truncated:
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )
            action_new = get_eps_greedy_action(Q[observation_new])
            updated_q_value = Q[observation, action] + ALPHA * (
                reward - Q[observation, action]
            )
            for a in range(action_space.n):
                updated_q_value += ALPHA * _get_action_prob(Q) * Q[observation_new, a]
            Q[observation, action] = updated_q_value
            observation = observation_new
            action = action_new

        pi = get_policy(Q, observation_space)
        if success_cb(pi, step):
            return True, pi, step

        cur_episode_len += 1
        if cur_episode_len > 100:
            break

    return False, get_policy(Q, observation_space), step


@with_default_values
def double_q(
    env, success_cb: Callable[[np.ndarray, int], bool], max_steps: int
) -> tuple[bool, np.ndarray, int]:
    observation_space, action_space = get_observation_action_space(env)
    Q_1 = np.zeros((observation_space.n, action_space.n))
    Q_2 = np.zeros((observation_space.n, action_space.n))

    for step in range(max_steps):
        observation, _ = env.env.reset()

        terminated = truncated = False

        while not terminated and not truncated:
            action = get_eps_greedy_action(Q_1[observation], env.eps(step))
            observation_new, reward, terminated, truncated, _ = env.step(
                action, observation
            )

            if random.randint(0, 100) < 50:
                Q_1[observation, action] = Q_1[observation, action] + ALPHA * (
                    reward
                    + env.gamma * Q_2[observation_new, np.argmax(Q_1[observation_new])]
                    - Q_1[observation, action]
                )
            else:
                Q_2[observation, action] = Q_2[observation, action] + ALPHA * (
                    reward
                    + env.gamma * Q_1[observation_new, np.argmax(Q_2[observation_new])]
                    - Q_2[observation, action]
                )
            observation = observation_new

        pi = get_policy(Q_1, observation_space)
        if success_cb(pi, step):
            return True, pi, step

    return False, get_policy(Q_1, observation_space), step
