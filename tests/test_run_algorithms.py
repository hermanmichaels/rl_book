from dp import policy_iteration
from gym_utils import get_observation_action_space


def test_dp(env_train):
    observation_space, _ = get_observation_action_space(env_train.env)
    pi = policy_iteration(env_train)
    assert pi.shape == (observation_space.n,)