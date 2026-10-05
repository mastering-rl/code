from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def select_action(self, state, action):
        pass

    def set_stochastic(self, stochastic):
        self.stochastic = stochastic

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass


class DeterministicPolicy(Policy):

    @abstractmethod
    def update(self, state, action):
        pass

    def set_stochastic(self, stochastic):
        pass


class StochasticPolicy(Policy):
    @abstractmethod
    def update(self, states, actions, rewards):
        pass

    @abstractmethod
    def get_probability(self, state, action):
        pass
