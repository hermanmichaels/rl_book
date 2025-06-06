from abc import ABC
import copy

class Algorithm(ABC):
    def __init__(self, env):
        self.env = env

    def update(self, episode):
        pass

    def get_action(self):
        raise NotImplementedError

    def finalize(self, episode, step):
        pass

    def get_policy(self):
        raise NotImplementedError
    
    def clone(self):
        cloned = self.__class__()
        return cloned
