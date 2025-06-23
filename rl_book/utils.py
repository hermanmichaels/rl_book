from typing import Callable


class ConstantFactory:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value