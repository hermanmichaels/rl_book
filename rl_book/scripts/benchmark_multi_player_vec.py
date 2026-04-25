import argparse
from functools import partial

import torch
from pettingzoo.classic import connect_four_v3, tictactoe_v3

from rl_book.env import ConnectFourEnv, MultiPlayerEnv, TicTacToeEnv
from rl_book.methods.inference import test_against_user
from rl_book.methods.method import MethodWithStats
from rl_book.methods.misc import RandomBatched
from rl_book.methods.models import ConnectFourCNN, TicTacToeMLP
from rl_book.methods.td_approx import (SemiGradientSarsaCNN)
from rl_book.methods.training import train_multi_player_vectorized

torch.autograd.set_detect_anomaly(True)


def get_env(env_name: str, device: torch.device, render_mode=None):
    env: MultiPlayerEnv
    if env_name == "TicTacToe":
        env = TicTacToeEnv(
            tictactoe_v3.env(render_mode=render_mode), 0.95, device=device
        )
    elif env_name == "ConnectFour":
        env = ConnectFourEnv(
            connect_four_v3.env(render_mode=render_mode), 0.95, device=device
        )
    else:
        raise ValueError(f"Unknown method name {env_name}")

    return env


def benchmark_multi_player(
    env_name: str, load_weights: bool, device: torch.device
) -> None:
    env_fn = partial(get_env, env_name, device)
    env = env_fn()
    network_class = TicTacToeMLP if env_name == "TicTacToe" else ConnectFourCNN

    methods = [
        MethodWithStats(RandomBatched(env)),
        MethodWithStats(
            SemiGradientSarsaCNN(
                env,
                load_weights=load_weights,
                network_class=network_class,
                device=device,
            )
        ),
    ]
    zoo = [MethodWithStats(RandomBatched(env))]
    # Train given methods
    train_multi_player_vectorized(
        env_fn, methods, zoo, max_steps=1000000, plot_interval=1000
    )

    # Now give user chance to play against one of the methods
    # TOOD: need good wrapper from action to input
    env = get_env(env_name, device, "human")
    test_against_user(env, methods[-1].method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RL methods in multi-player setup"
    )
    parser.add_argument("--env", type=str, required=True, help="Env")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_multi_player(args.env, args.load, device)
