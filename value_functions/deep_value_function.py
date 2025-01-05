import torch
import torch.nn as nn
from torch.optim import Adam

from mastering_rl.value_functions.value_function import ValueFunction


class DeepValueFunction(ValueFunction):
    """
    A neural network to represent the Value-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(self, state_space, hidden_dim=64, alpha=0.001):
        # Create a sequential neural network to represent the Q function
        self.value_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )
        self.optimiser = Adam(self.value_network.parameters(), lr=alpha)

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.value_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.value_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    def update(self, state, delta):
        self.batch_update([state], [delta])

    def batch_update(self, states, deltas):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        values = self.value_network(states_tensor)
        
        # Construct the target values
        targets = [
            value + delta for value, delta in zip(values.squeeze(1).tolist(), deltas)
        ]
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
        loss = nn.functional.smooth_l1_loss(
            values, 
            targets_tensor.unsqueeze(1)
        )

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
        self.optimiser.step()
        return loss

    def get_value(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            value = self.value_network(state_tensor)
        return value.item()

    def get_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            values = self.value_network(states_tensor)
        return values.squeeze(1).tolist()
