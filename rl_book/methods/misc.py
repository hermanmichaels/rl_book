import random

from rl_book.methods.method import RLMethod


class Random(RLMethod):
    def __init__(self, env):
        super().__init__(env)

    def get_name(self) -> str:
        return "Random"

    def act(self, state, step, mask=None):
        allowed_actions = self.get_allowed_actions(mask)
        return random.choice(allowed_actions)
