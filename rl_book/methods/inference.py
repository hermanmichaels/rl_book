import numpy as np
import torch
from gymnasium.core import Env

from rl_book.env import MultiPlayerEnv
from rl_book.methods.method import RLMethod

NUM_STEPS = 1000


def test_single_player(env: Env, method: RLMethod) -> None:
    method.eval()

    observation, _ = env.reset()

    for _ in range(NUM_STEPS):
        action = method.act(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_against_user(env: MultiPlayerEnv, method: RLMethod) -> None:
    env.env.reset()

    for agent in env.env.agent_iter():  # type: ignore
        (
            observation,
            _,
            termination,
            truncation,
            _,
        ) = env.env.last()  # type: ignore

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]

            state = env.obs_to_state(observation["observation"], 0, method.obs_mode)
            if agent == "player_1":
                # TODO: quick hack to make approximate methods work here
                if method.get_name() == "SemiGradientSarsa-CNN":
                    action = method.act(
                        state.unsqueeze(0), mask=np.expand_dims(mask, 0)
                    )
                else:
                    action = method.act(state, mask=mask)
            else:
                action = int(input(env.user_query()))

        env.env.step(action.item() if isinstance(action, torch.Tensor) else action)

    env.env.close()
