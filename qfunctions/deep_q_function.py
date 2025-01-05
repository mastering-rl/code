import random
import torch
import torch.nn as nn
from torch.optim import Adam

from mastering_rl.qfunctions.qfunction import QFunction


class DeepQFunction(QFunction):
    """A neural network to represent the Q-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(self, state_space, action_space, hidden_dim=128, alpha=0.001):

        # Create a sequential neural network to represent the Q function
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_space),
        )
        self.optimiser = Adam(self.q_network.parameters(), lr=alpha, amsgrad=True)

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Ensure the last layer outputs logits close to zero
        last_layer = self.q_network[-1]
        if isinstance(last_layer, nn.Linear):
            with torch.no_grad():
                last_layer.weight.fill_(0)
                last_layer.bias.fill_(0)

    def update(self, state, action, delta):
        return self.batch_update([state], [action], [delta])

    def batch_update(self, experiences):
        (states, actions, deltas, dones) = zip(*experiences)
        return self.batch_update(states, actions, deltas)

    def batch_update(self, states, actions, deltas):
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        q_values = (
            self.q_network(states_tensor)
            .gather(dim=1, index=actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # Construct the target values
        targets = [value + delta for value, delta in zip(q_values.tolist(), deltas)]
        targets_tensor = torch.as_tensor(targets, dtype=torch.float32)

        loss = nn.functional.smooth_l1_loss(
            q_values,
            targets_tensor,
        ).sum()

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimiser.step()
        return loss

    def get_q_values(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        with torch.no_grad():
            q_values = self.q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            )
        return q_values.squeeze(1).tolist()

    def get_max_q_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            max_q_values = self.q_network(states_tensor).max(1).values
        return max_q_values.tolist()

    def get_q_value(self, state, action):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        q_value = q_values[action].item()

        return q_value

    def get_max_pair(self, state, actions):
        # Convert the state into a tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        max_q = float("-inf")
        max_actions = []
        for action in actions:
            q_value = q_values[action].item()
            if q_value > max_q:
                max_actions = [action]
                max_q = q_value
            elif q_value == max_q:
                max_actions += [action]

        arg_max_q = random.choice(max_actions)
        return (arg_max_q, max_q)

    def soft_update(self, policy_qfunction, tau=0.005):
        target_dict = self.q_network.state_dict()
        policy_dict = policy_qfunction.q_network.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.q_network.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
