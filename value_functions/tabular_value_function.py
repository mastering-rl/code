from collections import defaultdict

from mastering_rl.value_functions.value_function import ValueFunction


class TabularValueFunction(ValueFunction):
    def __init__(self, alpha=0.1, default=0.0):
        self.value_table = defaultdict(lambda: default)
        self.alpha = alpha

    def add(self, state, value):
        self.value_table[state] = value

    def update(self, state, delta):
        self.value_table[state] += self.alpha * delta

    def batch_update(self, states, deltas):
        for state, delta in zip(states, deltas):
            self.update(state, delta)

    def merge(self, value_table):
        for state in value_table.value_table.keys():
            self.add(state, value_table.get_value(state))

    def get_value(self, state):
        return self.value_table[state]

    def get_values(self, states):
        return [self.get_value(state) for state in states]

