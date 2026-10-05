import json
from collections import defaultdict

from mastering_rl.qfunctions.qfunction import QFunction


class QTable(QFunction):
    def __init__(self, alpha=0.1, default_q_value=0.0):
        self.default_q_value = default_q_value
        self.qtable = defaultdict(lambda: self.default_q_value)
        self.alpha = alpha

    def update(self, state, action, delta):
        self.qtable[(state, action)] = self.qtable[(state, action)] + self.alpha * delta

    def batch_update(self, states, actions, deltas):
        for state, action, delta in zip(states, actions, deltas):
            self.update(state, action, delta)

    def soft_update(self, policy_qfunction, tau=0.01):
        for (state, action), q_value in policy_qfunction.qtable.items():
            self.qtable[(state, action)] = (
                tau * q_value + (1 - tau) * self.qtable[(state, action)]
            )

    def get_q_value(self, state, action):
        return self.qtable[(state, action)]

    def get_q_values(self, states, actions):
        return [
            self.get_q_value(state, action) for state, action in zip(states, actions)
        ]

    def reset(self):
        self.qtable = defaultdict(lambda: self.default_q_value)

    def save(self, filename):
        with open(filename, "w") as file:
            serialised = {str(key): value for key, value in self.qtable.items()}
            json.dump(serialised, file)

    def load(self, filename, default=0.0):
        with open(filename, "r") as file:
            serialised = json.load(file)
            self.qtable = defaultdict(
                lambda: default,
                {tuple(eval(key)): value for key, value in serialised.items()},
            )
