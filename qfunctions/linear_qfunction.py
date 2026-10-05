import json

from mastering_rl.qfunctions.qfunction import QFunction


class LinearQFunction(QFunction):
    def __init__(self, features, alpha=0.1, weights=None, default_q_value=0.0):
        self.features = features
        self.alpha = alpha
        self.default_q_value = default_q_value
        if weights == None:
            self._initialize_weights()
        else:
            self.weights = weights

    def _initialize_weights(self):
        self.weights = [
            self.default_q_value
            for _ in range(0, self.features.num_actions())
            for _ in range(0, self.features.num_features())
        ]

    def update(self, state, action, delta):
        # update the weights
        feature_values = self.features.extract(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (self.alpha * delta * feature_values[i])

    def get_q_value(self, state, action):
        q_value = 0.0
        feature_values = self.features.extract(state, action)
        for i in range(len(feature_values)):
            q_value += feature_values[i] * self.weights[i]
        return q_value

    def reset(self):
        self._initialize_weights()

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({'alpha': self.alpha, 'weights': self.weights}, f)

    @classmethod
    def load(cls, features, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls(features, alpha=data['alpha'], weights=data['weights'])

