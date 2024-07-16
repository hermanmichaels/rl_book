from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 1000


class Bandit:
    def __init__(self, mu: float, sigma: float = 1) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma, 1)


def initialize_bandits() -> list[Bandit]:
    return [
        Bandit(0.2),
        Bandit(-0.8),
        Bandit(1.5),
        Bandit(0.4),
        Bandit(1.1),
        Bandit(-1.5),
        Bandit(-0.1),
        Bandit(1),
        Bandit(0.7),
        Bandit(-0.5),
    ]


def simple_crit(Q: np.ndarray, N: np.ndarray, t: int, eps: float) -> int:
    return (
        int(np.argmax(Q))
        if np.random.rand() < 1 - eps
        else np.random.randint(Q.shape[0])
    )


def ucb_crit(Q: np.ndarray, N: np.ndarray, t: int, c: float) -> int:
    return int(np.argmax(Q + c * np.sqrt(np.log(t) / N)))


def bandit_solver(
    bandits: list[Bandit], crit: Callable[[np.ndarray, np.ndarray, int], int]
) -> np.ndarray:
    Q = np.zeros((len(bandits)))
    N = np.zeros((len(bandits)))

    rewards = []
    for t in range(NUM_STEPS):
        A = crit(Q, N, t)
        R = bandits[A]()
        rewards.append(R)
        N[A] = N[A] + 1
        Q[A] = Q[A] + 1 / N[A] * (R - Q[A])

    return np.cumsum(rewards) / np.arange(1, len(rewards) + 1)


def main() -> None:
    bandits = initialize_bandits()
    epss = [0, 0.01, 0.1]
    reward_averages = [bandit_solver(bandits, lambda q, n, t: simple_crit(q, n, t, eps)) for eps in epss]
    colors = ["r-", "b-", "g-"]
    for idx, reward_average in enumerate(reward_averages):
        plt.plot(
            range(len(reward_average)), reward_average, colors[idx], label=epss[idx]
        )
    plt.legend()
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
