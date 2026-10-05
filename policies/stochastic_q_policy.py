from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.policy import StochasticPolicy

""" Make a stochastic policy from a qfunction and a mutli-armed bandit.
    This helps to avoid e.g. loops in policies.
    This policy cannot be updated -- it is only for execution.
"""


class StochasticQPolicy(StochasticPolicy):
    def __init__(self, qfunction, bandit=EpsilonGreedy(epsilon=0.05)):
        self.qfunction = qfunction
        self.bandit = bandit

    def select_action(self, state, actions):
        return self.bandit.select(state, actions, self.qfunction)

    def update(self, states, actions, deltas):
        self.qfunction.batch_update(states, actions, deltas)

    def save(self, filename):
        self.qfunction.save(filename)

    def get_probability(self, state, action):
        raise NotImplementedError("get_probability not implemented for StochasticQPolicy.")

    @staticmethod
    def load(qfunction, filename):
        raise NotImplementedError("Loading a StochasticQPolicy is not implemented. Load the qfunction and create a StochasticQPolicy.")