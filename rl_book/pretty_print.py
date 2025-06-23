import sys

from rl_book.methods.method import MethodWithStats

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log_methods(methods: list[MethodWithStats], step: int) -> None:
    sorted_methods = sorted(methods, key=lambda x: -x.get_win_ratio())
    max_name_len = max(len(m.method.get_name()) for m in methods)
    separator = "-" * (max_name_len) + "|" + ("-" * 26)

    if step == 0:
        title = f"{BOLD}=== Method Stats at Step {step} ==={RESET}"
        legend = (
            f"{'Method '.ljust(max_name_len)}| {GREEN} Wins{RESET}"
            f"/ {YELLOW} Draws{RESET} / {RED} Losses{RESET}"
        )
        print(title)
        print(legend)
        print(separator)
        for i in range(len(methods)):
            print(f"row {i}| initializing...")

    sys.stdout.write(f"\033[{len(methods)}A")

    for method in sorted_methods:
        name = method.method.get_name().ljust(max_name_len)
        win = f"{method.get_win_ratio()*100:5.1f}%"
        draw = f"{method.get_draw_ratio()*100:5.1f}%"
        loss = f"{method.get_loss_ratio()*100:5.1f}%"

        line = (
            f"\033[K{name}| "
            f"{GREEN}{win}{RESET} / "
            f"{YELLOW}{draw}{RESET} / "
            f"{RED}{loss}{RESET}\n"
        )

        sys.stdout.write(line)

    sys.stdout.flush()
