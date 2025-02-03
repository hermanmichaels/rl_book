import gymnasium as gym

from env import ParametrizedEnv
from planning import mcts

GAMMA = 0.9
EPS = 0.001
NUM_STEPS = 100


def solve_grid_world() -> None:
    """Solve the grid world problem using the chosen solving method.

    Args:
        method: solving method
    """
    gym_env_train = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
    )
    env_train = ParametrizedEnv(gym_env_train, GAMMA, EPS)

    gym_env_test = gym.make(
        "FrozenLake-v1",
        desc=None,
        map_name="4x4",
        is_slippery=False,
        render_mode="human",
    )

    # Test policy and visualize found solution
    gym_env_test.reset()
    actions: list[int] = []
    for _ in range(NUM_STEPS):
        action = mcts(env_train, actions)
        actions.append(action)
        _, _, terminated, truncated, _ = gym_env_test.step(action)
        if terminated or truncated:
            break
    gym_env_test.close()


if __name__ == "__main__":
    solve_grid_world()
