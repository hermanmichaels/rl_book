from dataclasses import dataclass


@dataclass
class ReplayItem:
    state: int
    action: int
    reward: float
    mask: bool
