from abc import ABC, abstractmethod


class ModelFreeLearner(ABC):
    @abstractmethod
    def execute(self, episodes=2000):
        pass