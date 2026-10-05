from PIL.ImagePalette import load
from mastering_rl.policies.policy import DeterministicPolicy
from mastering_rl.qfunctions.qtable import QTable

""" Make a deterministic policy from a value function.
    This policy cannot be updated -- it is only for execution.
"""


class ValuePolicy(DeterministicPolicy):
    def __init__(self, mdp, values):
        self.mdp = mdp
        self.values = values

    def select_action(self, state, actions):
        qfunction = QTable()
        for action in actions:
            q_value = self.values.get_q_value(self.mdp, state, action)
            qfunction.update(state, action, q_value)
        return qfunction.get_argmax_q(state, actions)

    ''' This policy is only for execution, it cannot be updated. '''
    def update(self, state, action):
        pass

    def load(self, filename):
        self.values.load(filename)

    def save(self, filename):
        self.values.save(filename)