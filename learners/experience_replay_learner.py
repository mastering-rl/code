import random
import numpy as np
from collections import deque, namedtuple
from itertools import count


from mastering_rl.learners.model_free_learner import ModelFreeLearner

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done", "delta")
)


class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque([], maxlen=buffer_size)

    def push(self, state, action, next_state, reward, done, delta):
        self.buffer.append(Transition(state, action, next_state, reward, done, delta))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def update_priorities(self, errors):
        pass

    def __len__(self):
        return len(self.buffer)

class ExperienceReplayLearner(ModelFreeLearner):
    def __init__(
        self,
        mdp,
        bandit,
        policy_qfunction,
        target_qfunction,
        buffer=None,
        batch_size=64,
        buffer_size=10000,
        update_period=4,
        min_replay_size=1000,
        target_update_tau=0.01,
    ):
        self.mdp = mdp
        self.bandit = bandit
        self.policy_qfunction = policy_qfunction
        self.target_qfunction = target_qfunction
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size=buffer_size) if buffer is None else buffer
        self.update_period = update_period
        self.min_replay_size = min_replay_size
        self.target_update_tau = target_update_tau

        # Initialize target network by copying the policy network.
        self.target_qfunction.soft_update(
            self.policy_qfunction, tau=1.0
        )

    def execute(self, episodes=100, max_episode_length=float("inf")):

        episode_rewards = []
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            episode_reward = 0.0
            for step in count():
                actions = self.mdp.get_actions(state)
                action = self.bandit.select(state, actions, self.policy_qfunction)
                (next_state, reward, done) = self.mdp.execute(state, action)
                reached_episode_limit = step + 1 >= max_episode_length
                terminal = done or reached_episode_limit

                delta = self.get_delta(reward, state, action, next_state, terminal)
                self.buffer.push(state, action, next_state, reward, terminal, delta)

                # Perform an update on the policy qfunction using a batch
                if len(self.buffer) >= max(self.batch_size, self.min_replay_size):
                    transitions = self.buffer.sample(self.batch_size)
                    batch = Transition(*zip(*transitions))

                    deltas = self.get_deltas(
                        batch.reward,
                        batch.state,
                        batch.action,
                        batch.next_state,
                        batch.done,
                    )

                    self.policy_qfunction.batch_update(
                        batch.state, batch.action, deltas
                    )

                    self.buffer.update_priorities(deltas)

                if (
                    len(self.buffer) >= max(self.batch_size, self.min_replay_size)
                    and (step + 1) % self.update_period == 0
                ):
                    # Soft update of the target Q-function
                    self.target_qfunction.soft_update(
                        self.policy_qfunction, tau=self.target_update_tau
                    )

                # Move to the next state
                state = next_state
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if terminal:
                    break

            episode_rewards.append(episode_reward)
        return episode_rewards

    """ Calculate the deltas for the update """

    def get_deltas(self, rewards, states, actions, next_states, dones):
        q_values = self.policy_qfunction.get_q_values(states, actions)

        # Double DQN target: policy network selects next actions, target network evaluates them.
        next_actions = [
            self.policy_qfunction.get_argmax_q(next_state, self.mdp.get_actions(next_state))
            for next_state in next_states
        ]
        next_state_q_values = self.target_qfunction.get_q_values(
            next_states, next_actions
        )
        deltas = [
            (
                reward + (self.mdp.get_discount_factor() * next_state_q_value) - q_value
                if not done
                else (reward - q_value)
            )
            for reward, next_state_q_value, q_value, done in zip(
                rewards, next_state_q_values, q_values, dones
            )
        ]
        return deltas

    def get_delta(self, reward, state, action, next_state, done):
        q_value = self.policy_qfunction.get_q_value(state, action)
        next_action = self.policy_qfunction.get_argmax_q(
            next_state, self.mdp.get_actions(next_state)
        )
        next_state_q_value = self.target_qfunction.get_q_value(next_state, next_action)
        deltas = (
            reward + (self.mdp.get_discount_factor() * next_state_q_value) - q_value
            if not done
            else (reward - q_value)
        )
        return deltas
