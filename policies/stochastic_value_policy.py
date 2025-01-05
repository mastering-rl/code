from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.policy import StochasticPolicy
from mastering_rl.qfunctions.qtable import QTable

""" Make a deterministic policy from a value function.
    This policy cannot be updated -- it is only for execution.
"""


class StochasticValuePolicy(StochasticPolicy):
    def __init__(self, mdp, values, bandit=EpsilonGreedy(epsilon=0.05)):
        self.mdp = mdp
        self.values = values
        self.bandit = bandit

    def select_action(self, state, actions):
        qfunction = QTable()
        for action in actions:
            q_value = self.values.get_q_value(self.mdp, state, action)
            qfunction.update(state, action, q_value)
        return self.bandit.select(state, actions, qfunction)
