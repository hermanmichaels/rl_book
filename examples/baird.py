from typing import Any
import matplotlib.pyplot as plt
import numpy as np


class BairdCounterexample:
    """
    A small Baird-style 7-state off-policy prediction problem.

    States:
        0..5 : upper states
        6    : bottom state

    Actions:
        0 = dashed ("behavior-favored"): next state is a random upper state
        1 = solid  ("target"):           next state is the bottom state

    Behavior policy:
        mu(dashed) = 6/7
        mu(solid)  = 1/7

    Target policy:
        pi(solid)  = 1

    Rewards are always zero, so the true value function is zero everywhere.

    Features are the standard 8-dimensional features commonly used for
    Baird's counterexample.
    """

    def __init__(self, gamma: float = 0.99, seed: int = 0) -> None:
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.n_states = 7
        self.n_features = 8
        self.upper_states = np.arange(6)
        self.bottom_state = 6

        self.features = np.zeros((7, 8), dtype=np.float64)

        # Upper states: own coordinate gets 2, last coordinate gets 1
        for s in range(6):
            self.features[s, s] = 2.0
            self.features[s, 7] = 1.0

        # Bottom state: coordinate 6 gets 1, last coordinate gets 2
        self.features[6, 6] = 1.0
        self.features[6, 7] = 2.0

    def sample_state(self) -> int:
        # Use an i.i.d. sampling distribution over states for a clean demo.
        return int(self.rng.integers(0, self.n_states))

    def sample_behavior_action(self) -> int:
        # 0 = dashed, 1 = solid
        return 0 if self.rng.random() < (6.0 / 7.0) else 1

    def importance_ratio(self, action: int) -> float:
        # pi(solid|s)=1, pi(dashed|s)=0
        # mu(solid|s)=1/7, mu(dashed|s)=6/7
        if action == 0:
            return 0.0
        return 7.0

    def sample_next_state(self, state: int, action: int) -> int:
        if action == 0:  # dashed
            return int(self.rng.choice(self.upper_states))
        return self.bottom_state  # solid

    def reward(self, state: int, action: int, next_state: int) -> float:
        del state, action, next_state
        return 0.0


def value_hat(phi: np.ndarray, w: np.ndarray) -> float:
    return float(phi @ w)


def rms_value_error(env: BairdCounterexample, w: np.ndarray) -> float:
    # True value is zero everywhere
    preds = env.features @ w
    return float(np.sqrt(np.mean(preds**2)))


def run_td(
    env: BairdCounterexample,
    alpha: float,
    num_steps: int,
    seed: int,
):
    """Off-policy TD(0) with importance sampling (using semi-gradient TD)."""
    env = BairdCounterexample(gamma=env.gamma, seed=seed)
    w = np.ones(env.n_features, dtype=np.float64)
    w[6] = 10.0  # classic asymmetric initialization

    logs: dict[str, list[np.ndarray | float]] = {"weight_norm": [], "rmsve": [], "values": []}

    for t in range(num_steps):
        s = env.sample_state()
        a = env.sample_behavior_action()
        rho = env.importance_ratio(a)

        s_next = env.sample_next_state(s, a)
        r = env.reward(s, a, s_next)

        phi = env.features[s]
        phi_next = env.features[s_next]

        delta = r + env.gamma * value_hat(phi_next, w) - value_hat(phi, w)

        # Semi-gradient off-policy TD
        w += alpha * rho * delta * phi

        if t % 100 == 0:
            logs["weight_norm"].append(float(np.linalg.norm(w)))
            logs["rmsve"].append(rms_value_error(env, w))
            logs["values"].append(env.features @ w)

    return w, logs


def run_residual_gradient(
    env: BairdCounterexample,
    alpha: float,
    num_steps: int,
    seed: int,
):
    """
    True residual-gradient style update with double sampling.

    We sample two independent next states from the same (s, a):
      - one for the TD residual delta
      - one for the gradient term

    This avoids the double-sampling issue for the demo because we control
    the simulator.
    """
    env = BairdCounterexample(gamma=env.gamma, seed=seed)
    w = np.ones(env.n_features, dtype=np.float64)
    w[6] = 10.0

    logs: dict[str, list[np.ndarray | float]] = {"weight_norm": [], "rmsve": [], "values": []}

    for t in range(num_steps):
        s = env.sample_state()
        a = env.sample_behavior_action()
        rho = env.importance_ratio(a)

        s_next_1 = env.sample_next_state(s, a)
        s_next_2 = env.sample_next_state(s, a)
        r = env.reward(s, a, s_next_1)

        phi = env.features[s]
        phi_next_1 = env.features[s_next_1]
        phi_next_2 = env.features[s_next_2]

        delta = r + env.gamma * value_hat(phi_next_1, w) - value_hat(phi, w)

        # Residual-gradient update:
        # w <- w + alpha * rho * delta * (phi - gamma * phi_next_2)
        w += alpha * rho * delta * (phi - env.gamma * phi_next_2)

        if t % 100 == 0:
            logs["weight_norm"].append(float(np.linalg.norm(w)))
            logs["rmsve"].append(rms_value_error(env, w))
            logs["values"].append(env.features @ w)

    return w, logs


def run_gtd2(
    env: BairdCounterexample,
    alpha: float,
    beta: float,
    num_steps: int,
    seed: int,
):
    """
    Off-policy GTD2-style implementation using importance sampling.

    Book/paper form for GTD2:
        theta_{k+1} = theta_k + alpha (phi - gamma phi') (phi^T w)
        w_{k+1}     = w_k + beta (delta - phi^T w) phi

    Here we apply the importance ratio rho to the off-policy sample terms.
    """
    env = BairdCounterexample(gamma=env.gamma, seed=seed)
    theta = np.ones(env.n_features, dtype=np.float64)
    theta[6] = 10.0
    aux = np.zeros(env.n_features, dtype=np.float64)

    logs: dict[str, list[np.ndarray | float]] = {"weight_norm": [], "rmsve": [], "values": []}

    for t in range(num_steps):
        s = env.sample_state()
        a = env.sample_behavior_action()
        rho = env.importance_ratio(a)

        s_next = env.sample_next_state(s, a)
        r = env.reward(s, a, s_next)

        phi = env.features[s]
        phi_next = env.features[s_next]

        delta = r + env.gamma * value_hat(phi_next, theta) - value_hat(phi, theta)

        theta += alpha * rho * (phi - env.gamma * phi_next) * (phi @ aux)
        aux += beta * ((rho * delta) - (phi @ aux)) * phi

        if t % 100 == 0:
            logs["weight_norm"].append(float(np.linalg.norm(theta)))
            logs["rmsve"].append(rms_value_error(env, theta))
            logs["values"].append(env.features @ theta)

    return theta, aux, logs


def run_tdc(
    env: BairdCounterexample,
    alpha: float,
    beta: float,
    num_steps: int,
    seed: int,
):
    """
    Off-policy TDC-style implementation using importance sampling.

    Paper form for TDC:
        theta_{k+1} = theta_k + alpha delta phi - alpha gamma phi' (phi^T w)
        w_{k+1}     = w_k + beta (delta - phi^T w) phi

    Here we apply the importance ratio rho to the off-policy sample terms.
    """
    env = BairdCounterexample(gamma=env.gamma, seed=seed)
    theta = np.ones(env.n_features, dtype=np.float64)
    theta[6] = 10.0
    aux = np.zeros(env.n_features, dtype=np.float64)

    logs: dict[str, list[np.ndarray | float]] = {"weight_norm": [], "rmsve": [], "values": []}

    for t in range(num_steps):
        s = env.sample_state()
        a = env.sample_behavior_action()
        rho = env.importance_ratio(a)

        s_next = env.sample_next_state(s, a)
        r = env.reward(s, a, s_next)

        phi = env.features[s]
        phi_next = env.features[s_next]

        delta = r + env.gamma * value_hat(phi_next, theta) - value_hat(phi, theta)

        theta += alpha * rho * (delta * phi - env.gamma * (phi @ aux) * phi_next)
        aux += beta * ((rho * delta) - (phi @ aux)) * phi

        if t % 100 == 0:
            logs["weight_norm"].append(float(np.linalg.norm(theta)))
            logs["rmsve"].append(rms_value_error(env, theta))
            logs["values"].append(env.features @ theta)

    return theta, aux, logs


def plot_results(results: dict[str, dict[str, list[float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for name, logs in results.items():
        if name == "TD":
            continue
        axes[0].plot(logs["weight_norm"], label=name)
    axes[0].set_title("Weight norm")
    axes[0].set_xlabel("Logged step (x100)")
    axes[0].set_ylabel(r"$\|w\|_2$")
    axes[0].set_yscale("log")
    axes[0].legend()

    for name, logs in results.items():
        if name == "TD":
            continue
        axes[1].plot(logs["rmsve"], label=name)
    axes[1].set_title("RMS value error")
    axes[1].set_xlabel("Logged step (x100)")
    axes[1].set_ylabel("RMSVE")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main() -> None:
    env = BairdCounterexample(gamma=0.99, seed=0)

    num_steps = 100_000

    # These are demo hyperparameters, not a canonical standard.
    td_alpha = 0.005

    rg_alpha = 0.005

    gtd2_alpha = 0.005
    gtd2_beta = 0.05

    tdc_alpha = 0.005
    tdc_beta = 0.05

    td_w, td_logs = run_td(env, alpha=td_alpha, num_steps=num_steps, seed=0)
    rg_w, rg_logs = run_residual_gradient(
        env, alpha=rg_alpha, num_steps=num_steps, seed=0
    )
    gtd2_theta, gtd2_aux, gtd2_logs = run_gtd2(
        env,
        alpha=gtd2_alpha,
        beta=gtd2_beta,
        num_steps=num_steps,
        seed=0,
    )
    tdc_theta, tdc_aux, tdc_logs = run_tdc(
        env,
        alpha=tdc_alpha,
        beta=tdc_beta,
        num_steps=num_steps,
        seed=0,
    )

    print("Final TD weights:")
    print(np.round(td_w, 4))
    print()

    print("Final Residual Gradient weights:")
    print(np.round(rg_w, 4))
    print()

    print("Final GTD2 theta:")
    print(np.round(gtd2_theta, 4))
    print("Final GTD2 auxiliary weights:")
    print(np.round(gtd2_aux, 4))
    print()

    print("Final TDC theta:")
    print(np.round(tdc_theta, 4))
    print("Final TDC auxiliary weights:")
    print(np.round(tdc_aux, 4))
    print()

    results = {
        "TD": td_logs,
        "Residual Gradient": rg_logs,
        "GTD2": gtd2_logs,
        "TDC": tdc_logs,
    }
    plot_results(results)


if __name__ == "__main__":
    main()
