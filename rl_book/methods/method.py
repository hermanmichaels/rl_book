from abc import ABC

class Algorithm(ABC):
    def update(self):
        pass

    def get_action(self):
        pass

    def finalize(self, episode):
        pass
