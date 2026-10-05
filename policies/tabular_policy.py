from collections import defaultdict

from mastering_rl.policies.policy import DeterministicPolicy


class TabularPolicy(DeterministicPolicy):
    def __init__(self, default_action=None):
        self.policy_table = defaultdict(lambda: default_action)

    def select_action(self, state, actions):
        return self.policy_table[state]

    def update(self, state, action):
        self.policy_table[state] = action

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.policy_table, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            policy_table = json.load(f)
        return cls(policy_table=policy_table)