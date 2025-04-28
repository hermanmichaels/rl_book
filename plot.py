import matplotlib.pyplot as plt

def plot_results(needed_steps, fig_path):
    x_values = [10, 20, 30, 40, 50]
    markers = ["o", "s", "^", "*"]

    labels = ["value_iteration", "on_policy_mc", "q", "sarsa_n", "dyna_q"]
    for idx, y_values in enumerate(needed_steps):
        plt.plot(
            x_values,
            y_values,
            marker=markers[idx % len(markers)],
            label=labels[idx],
        )
        plt.legend()
        plt.xticks([10, 20, 30, 40, 50])
        plt.xlabel("Gridworld size")
        plt.ylabel("Steps needed")
        plt.savefig(fig_path)
    plt.clf()

value = [100, 100, 100, 100, 100]
on_policy_mc = [100, 900, 2000, 7000, 200000]
q = [1000, 5000, 200000, 100000, 200000]
sarsa_n = [800, 200000, 190000, 200000, 200000]
dyna_q = [800, 70000, 30000, 200000, 200000]
plot_results([value, on_policy_mc, q, sarsa_n, dyna_q], "results/all_new.png")

value = [0.5, 1.8, 2.7, 4.9, 7.9]
on_policy_mc = [0.02, 0.45, 4.04, 13.7, 500]
q = [0.8, 9.7, 500, 317, 500]
sarsa_n = [0.6, 500, 500, 500, 500]
dyna_q = [0.4, 127.4, 86, 500, 500]
plot_results([value, on_policy_mc, q, sarsa_n, dyna_q], "results/all_times.png")