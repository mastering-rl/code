from mastering_rl.policies.policy import DeterministicPolicy

""" Make a deterministic policy from a qfunction.
    This policy cannot be updated -- it is only for execution.
"""


class QPolicy(DeterministicPolicy):
    def __init__(self, qfunction):
        self.qfunction = qfunction

    def select_action(self, state, actions):
        return self.qfunction.get_argmax_q(state, actions)

    def update(self, state, action):
        return self.qfunction.update(state, action)

    def save(self, filename):
        self.qfunction.save(filename)

    def load(self, filename):
        self.qfunction.load(filename)
