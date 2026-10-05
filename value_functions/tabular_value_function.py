import json
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

    def load(self, filename):
        import json

        with open(filename, "r") as file:
            serialised = json.load(file)
            self.value_table = defaultdict(lambda: serialised[0])
            for key, value in serialised.items():
                try:
                    parsed_key = eval(key)
                except Exception:
                    parsed_key = key
                self.value_table[parsed_key] = value

    def save(self, filename):
        with open(filename, "w") as file:
            serialised = {
                str(key): value for key, value in self.value_table.items()
            }
            json.dump(serialised, file)

    def load(self, filename, default=0.0):
        with open(filename, "r") as file:
            serialised = json.load(file)
            self.value_table = defaultdict(
                lambda: default,
                {tuple(eval(key)): value for key, value in serialised.items()},
            )

    