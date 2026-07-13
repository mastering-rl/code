import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from mastering_rl.qfunctions.qfunction import QFunction


class DeepQFunction(QFunction):

    """A neural network to represent a Q-function.
    This class uses PyTorch for the neural network framework (https://pytorch.org/).
    """

    def __init__(self, state_space, action_space, hidden_dim=128, alpha=0.001, clip_max_norm=10, device='cpu'):

        # Assign the provided device to q_network
        self.device = device
        self.q_network = nn.Sequential(
            nn.Linear(in_features=state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_space),
        ).to(device)
        self.optimiser = Adam(self.q_network.parameters(), lr=alpha)
        self.clip_max_norm = clip_max_norm

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.q_network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def to(self, device):
        self.q_network.to(device)
        return self

    def update(self, state, action, delta):
        return self.batch_update([state], [action], [delta])

    def batch_update_from_experiences(self, experiences):
        (states, actions, deltas, dones) = zip(*experiences)
        return self.batch_update(states, actions, deltas)

    def batch_update(self, states, actions, deltas):
        # Materialize dense arrays first to avoid slow tensor creation from lists of ndarrays.
        states_tensor = torch.as_tensor(np.asarray(states, dtype=np.float32)).to(self.device)
        actions_tensor = torch.as_tensor(np.asarray(actions, dtype=np.int64)).to(self.device)
        deltas_tensor = torch.as_tensor(np.asarray(deltas, dtype=np.float32)).to(self.device)

        q_values = (
            self.q_network(states_tensor)
            .gather(dim=1, index=actions_tensor.unsqueeze(1))
            .squeeze(1)
        )

        # DQN target is current Q estimate plus TD error (delta).
        targets_tensor = q_values.detach() + deltas_tensor

        loss = nn.functional.mse_loss(
            q_values,
            targets_tensor,
        )
        
        self.optimiser.zero_grad()
        loss.backward()
        if self.clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=self.clip_max_norm)
        self.optimiser.step()
        return loss

    def get_q_values(self, states, actions):
        states_tensor = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(states_tensor).gather(
                1, actions_tensor.unsqueeze(1)
            )
        return q_values.squeeze(1).tolist()

    def get_max_q_values(self, states):
        states_tensor = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            max_q_values = self.q_network(states_tensor).max(1).values
        return max_q_values.tolist()
    
    def get_max_q_value(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            max_q_value = self.q_network(state_tensor).max().item()
        return max_q_value

    def get_q_value(self, state, action):
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        q_value = q_values[action].item()

        return q_value

    def get_max_pair(self, state, actions):
        # Convert the state into a tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Deterministic tie-breaking (first max) avoids extra randomness during early learning.
        best_action = actions[0]
        best_q = q_values[best_action].item()
        for action in actions[1:]:
            q_value = q_values[action].item()
            if q_value > best_q:
                best_action = action
                best_q = q_value

        return (best_action, best_q)

    def soft_update(self, policy_qfunction, tau=0.01):
        target_dict = self.q_network.state_dict()
        policy_dict = policy_qfunction.q_network.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
        self.q_network.load_state_dict(target_dict)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
