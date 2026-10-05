from abc import ABC, abstractmethod
import random


class QFunction(ABC):
    """Update the Q-value of (state, action) by delta"""

    @abstractmethod
    def update(self, state, action, delta):
        pass

    """ Get a Q value for a given state-action pair """

    @abstractmethod
    def get_q_value(self, state, action):
        pass

    """ Save a Q-function to a specified filename """

    @abstractmethod
    def save(self, filename):
        pass

    """ Load a Q-function from a specified filename """

    @abstractmethod
    def load(self, filename):
        pass

    """ Return the action with the maximum Q-value """

    def get_argmax_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return argmax_q

    """ Return the maximum Q-value in this Q-function """

    def get_max_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return max_q

    """ Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    """

    def get_max_pair(self, state, actions):
        max_q = float("-inf")
        max_actions = []
        for action in actions:
            value = self.get_q_value(state, action)
            if value > max_q:
                max_actions = [action]
                max_q = value
            elif value == max_q:
                max_actions += [action]

        arg_max_q = random.choice(max_actions)
        return (arg_max_q, max_q)
