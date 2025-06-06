from collections import defaultdict
from pettingzoo.classic import tictactoe_v3, connect_four_v3
from rl_book.gym_utils import get_observation_action_space
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from rl_book.methods.mc import McEs
from rl_book.methods.td import MinMax, QLearning, Random, Sarsa
from rl_book.methods.td_n import SarsaN
from rl_book.replay_utils import ReplayItem

ALPHA = 0.01
GAMMA = 0.97

class MethodStats:
    def __init__(self, method):
        self.method = method
        self.wins = 0
        self.picks = 0

    def update_pick(self):
        self.picks += 1

    def update_win(self,):
        self.wins += 1

    def get_win_ratio(self):
        return self.wins / (self.picks + 1)
    
    def clone(self):
        cloned = MethodStats(self.method.clone())
        cloned.wins = self.wins
        cloned.picks = self.picks
        return cloned

    
    # TODO: call

# def obs_to_state(obs, start_pos):
#     board = obs  # shape: (3, 3, 2)
#     state_flat = []

#     for row in range(3):
#         for col in range(3):
#             if board[row][col][0] == 1:
#                 state_flat.append(1)  # player 1
#             elif board[row][col][1] == 1:
#                 state_flat.append(2)  # player 2
#             else:
#                 state_flat.append(0)  # empty

#     state_flat.append(start_pos)

#     # Convert base-3 list to integer
#     state = 0
#     for i, val in enumerate(state_flat):
#         state += val * (3 ** i)
#     return state
    
def obs_to_state(obs, start_pos=None):
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

    if start_pos is not None:
        state_flat.append(start_pos)

    # Convert to base-3 integer
    state = 0
    for i, val in enumerate(state_flat):
        state += val * (3 ** i)

    return state

# MethodStats(MinMax()), 
methods = [MethodStats(Random()), MethodStats(QLearning()), MethodStats(Sarsa()), MethodStats(SarsaN()), MethodStats(McEs())]
zoo = [MethodStats(Random()), MethodStats(MinMax())]

NUM_EPOCHS = 100000
# PLAYERS = ["player_1", "player_2"]
PLAYERS = ["player_0", "player_1"]

win_ratios = [[(0, method.get_win_ratio())] for method in methods]

for epoch in range(NUM_EPOCHS):
    env = connect_four_v3.env() # tictactoe_v3.env()
    env.reset()

    method_idx = random.randint(0, len(methods) - 1)
    player_pos = random.randint(0, 1)

    opponent_idx = random.randint(0, len(zoo) - 1)
    opponent = zoo[opponent_idx]

    methods[method_idx].update_pick()
    zoo[opponent_idx].update_pick()

    state_dict = {}
    done = False

    episode = []

    while not done:
        agent = env.agent_selection
        # print(agent)
        observation, reward, termination, truncation, info = env.last()
        reward += 0.1

        done = termination or truncation

        if done:
            action = None

            # Game over, rewards contains all
            if env.rewards[PLAYERS[player_pos]] == 1:
                methods[method_idx].update_win()
            elif env.rewards[PLAYERS[1 - player_pos]] == 1:
                zoo[opponent_idx].update_win()
            
        else:
            mask = observation["action_mask"]
            state = obs_to_state(observation["observation"], player_pos)
            if agent == PLAYERS[player_pos]:
                action = methods[method_idx].method.act(state, mask)
            else:
                action = opponent.method.act(state, mask)
            state_dict[agent] = (state, action, mask)

        env.step(action)

        _, reward, _, _, _ = env.last()
        
        if (env.agent_selection == PLAYERS[player_pos] and env.agent_selection in state_dict):
            s, a, mask = state_dict[PLAYERS[player_pos]]
   
            observation_new = env.observe(PLAYERS[player_pos])
            mask_new = observation_new["action_mask"]

            episode.append(ReplayItem(s, a, reward, mask))

            if reward == 0:
                legal = []
            else:
                legal = observation_new["action_mask"]

            methods[method_idx].method.update(episode)

    episode.append(ReplayItem(obs_to_state(observation_new["observation"], player_pos), -1, 0, []))

    # print("finalize")
    methods[method_idx].method.finalize(episode)


    # import ipdb
    # ipdb.set_trace()

    if epoch % 100 == 0:
        for idx, method in enumerate(methods):
            win_ratios[idx].append((epoch, method.get_win_ratio()))

        plt.clf()
        for idx in range(len(methods)):
            x = [epoch for epoch, _ in win_ratios[idx]]
            y = [win_ratio for _, win_ratio in win_ratios[idx]]
            plt.plot(x, y, label=idx)

        plt.legend()
        plt.savefig("wins.png")

    if epoch % 50 == 0:
        zoo.append(methods[method_idx].clone())
        zoo = sorted(zoo, key=lambda x: -x.get_win_ratio())
        # import ipdb
        # ipdb.set_trace()
        zoo = zoo[:50]

    env.close()

# env = tictactoe_v3.env(render_mode="human")
env = connect_four_v3.env(render_mode="human")
# connect_four_v3
env.reset()

# TODO: methods as class, e.g. q.update

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]

        state = obs_to_state(observation["observation"], 0)
        if agent == "player_1":
            action = methods[1].method.act(state, mask)
        else:
            # action = methods[0].method.act(state, mask)
            action = int(input("Enter your action (column index): "))


    env.step(action)
            
env.close()

