import argparse

from pettingzoo.classic import connect_four_v3, tictactoe_v3

from rl_book.env import ConnectFourEnv, TicTacToeEnv
from rl_book.methods.inference import test_again_user
from rl_book.methods.mc import OffPolicyMC, OnPolicyMC
from rl_book.methods.method import MethodWithStats
from rl_book.methods.misc import Random
from rl_book.methods.td import DoubleQ, ExpectedSarsa, QLearning, Sarsa
from rl_book.methods.td_n import SarsaN, TreeN
from rl_book.methods.training import train_multi_player


def get_env(env_name: str, render_mode=None):
    if env_name == "TicTacToe":
        env = TicTacToeEnv(tictactoe_v3.env(render_mode=render_mode), 0.95)
    elif env_name == "ConnectFour":
        env = ConnectFourEnv(connect_four_v3.env(render_mode=render_mode), 0.95)
    else:
        raise ValueError(f"Unknown method name {env_name}")

    return env


def benchmark_multi_player(env_name: str) -> None:
    env = get_env(env_name)

    methods = [
        MethodWithStats(Random(env)),
        MethodWithStats(OnPolicyMC(env)),
        MethodWithStats(OffPolicyMC(env)),
        MethodWithStats(QLearning(env)),
        MethodWithStats(Sarsa(env)),
        MethodWithStats(ExpectedSarsa(env)),
        MethodWithStats(DoubleQ(env)),
        MethodWithStats(SarsaN(env)),
        MethodWithStats(TreeN(env)),
    ]
    zoo = [MethodWithStats(Random(env))]
    # Train given methods
    train_multi_player(env, methods, zoo, max_steps=100000, plot_interval=100)

    # Now give user chance to play against one of the methods
    # TOOD: need good wrapper from action to input
    env = get_env(env_name, "human")
    test_again_user(env, methods[1].method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RL methods in multi-player setup"
    )
    parser.add_argument("--env", type=str, required=True, help="Env")
    args = parser.parse_args()
    benchmark_multi_player(args.env)
