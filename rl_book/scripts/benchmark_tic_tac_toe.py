from pettingzoo.classic import connect_four_v3, tictactoe_v3

from rl_book.env import ConnectFourEnv, MethodStats, TicTacToeEnv
from rl_book.methods.mc import OnPolicyMC
from rl_book.methods.td import QLearning, Sarsa
from rl_book.methods.td_n import SarsaN
from rl_book.methods.training import train_multi_player
from rl_book.replay_utils import ReplayItem

# env = ConnectFourEnv(connect_four_v3.env(), 0.95)
env = TicTacToeEnv(tictactoe_v3.env(), 0.95)

methods = [
    MethodStats(QLearning(env)),
    MethodStats(Sarsa(env)),
    MethodStats(SarsaN(env)),
]
zoo = [MethodStats(QLearning(env))]

train_multi_player(env, methods, zoo, max_steps=100)

# env = ConnectFourEnv(connect_four_v3.env(render_mode="human"))
env = TicTacToeEnv(tictactoe_v3.env(render_mode="human"))
env.env.reset()

for agent in env.env.agent_iter():
    observation, reward, termination, truncation, info = env.env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]

        state = env.obs_to_state(observation["observation"], 0)
        if agent == "player_1":
            action = methods[1].method.act(state, mask, 1000)
        else:
            # action = methods[0].method.act(state, mask)
            action = int(input("Enter your action (column index): "))

    env.env.step(action)

env.env.close()
