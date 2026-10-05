import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from mastering_rl.policies.policy import StochasticPolicy


class DeepNeuralNetworkPolicy(StochasticPolicy):
    """
    An implementation of a policy that uses a PyTorch (https://pytorch.org/)
    deep neural network to represent the underlying policy.
    """

    def __init__(
        self, state_space, action_space, hidden_dim=128, alpha=0.001, stochastic=True
    ):
        self.state_space = state_space
        self.action_space = action_space

        # Define the policy structure as a sequential neural network.
        self.policy_network = nn.Sequential(
            nn.Linear(in_features=self.state_space, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.action_space),
        )

        # Initialize weights using Xavier initialization and biases to zero
        self._initialize_weights()

        # The optimiser for the policy network, used to update policy weights
        self.optimiser = Adam(self.policy_network.parameters(), lr=alpha)

        # Whether to select an action stochastically or deterministically
        self.stochastic = stochastic

    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    """ Select an action using a forward pass through the network """

    def select_action(self, state, actions):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        masked_logits = self._mask_logits(action_logits, actions)
        if self.stochastic:
            # Sample an action according to the probability distribution
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
        else:
            # Choose the action with the highest probability
            action = torch.argmax(masked_logits)
        return action.item()

    """ Get the probability of an action being selected in a state """

    def get_probability(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logits = self.policy_network(state)

        # A softmax layer turns action logits into relative probabilities
        probabilities = F.softmax(input=action_logits, dim=-1).tolist()

        # Convert from a tensor encoding back to the action space
        return probabilities[action]

    def _mask_logits(self, action_logits, actions):
        mask = torch.full_like(action_logits, float("-inf"))
        mask[actions] = 0
        return action_logits + mask

    def evaluate_actions(self, states, actions, action_spaces=None):
        action_logits = self.policy_network(states)
        if action_spaces is not None:
            masked_logits = []
            for logits, valid_actions in zip(action_logits, action_spaces):
                masked_logits.append(self._mask_logits(logits, valid_actions))
            action_logits = torch.stack(masked_logits)

        action_distribution = Categorical(logits=action_logits)
        log_prob = action_distribution.log_prob(actions)
        return log_prob

    def update(self, states, actions, deltas, action_spaces=None, entropy_coeff=0.01):
        deltas_tensor = torch.as_tensor(deltas, dtype=torch.float32).view(-1)
        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long).view(-1)

        if deltas_tensor.numel() > 1:
            deltas_tensor = (deltas_tensor - deltas_tensor.mean()) / (
                deltas_tensor.std(unbiased=False) + 1e-8
            )
        deltas_tensor = torch.clamp(deltas_tensor, min=-10.0, max=10.0)

        # Single forward pass for both log-prob and entropy
        action_logits = self.policy_network(states_tensor)
        if action_spaces is not None:
            masked_logits = []
            for logits, valid_actions in zip(action_logits, action_spaces):
                masked_logits.append(self._mask_logits(logits, valid_actions))
            action_logits = torch.stack(masked_logits)

        dist = Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()

        loss = -(action_log_probs * deltas_tensor.detach()).mean() - entropy_coeff * entropy

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimiser.step()
        return loss

    def set_stochastic(self, stochastic):
        self.stochastic = stochastic

    def reset(self):
        self._initialize_weights()

    def save(self, filename):
        torch.save({
            'state_space': self.state_space,
            'action_space': self.action_space,
            'state_dict': self.policy_network.state_dict(),
        }, filename)

    @classmethod
    def load(cls, filename):
        checkpoint = torch.load(filename)
        policy = cls(checkpoint['state_space'], checkpoint['action_space'])
        policy.policy_network.load_state_dict(checkpoint['state_dict'])
        return policy
