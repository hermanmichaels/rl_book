from dp import policy_iteration, value_iteration
from env import ParametrizedEnv
from mc import mc_es, off_policy_mc, off_policy_mc_non_inc, on_policy_mc
from planning import mcts
from td import double_q, expected_sarsa, q, sarsa
from td_n import sarsa_n, tree_n    


def find_policy(env_train, method):
    if method == "policy_iteration":
        pi = policy_iteration(env_train)
    elif method == "value_iteration":
        pi = value_iteration(env_train)
    elif method == "mc_es":
        pi = mc_es(env_train)
    elif method == "on_policy_mc":
        pi = on_policy_mc(env_train)
    elif method == "off_policy_mc":
        pi = off_policy_mc(env_train)
    elif method == "off_policy_mc_non_inc":
        pi = off_policy_mc_non_inc(env_train)
    elif method == "sarsa":
        pi = sarsa(env_train)
    elif method == "q":
        pi = q(env_train)
    elif method == "expected_sarsa":
        pi = expected_sarsa(env_train)
    elif method == "double_q":
        pi = double_q(env_train)
    elif method == "sarsa_n":
        pi = sarsa_n(env_train)
    elif method == "tree_n":
        pi = tree_n(env_train)
    else:
        raise ValueError(f"Unknown solution method {method}")
                         
    return pi