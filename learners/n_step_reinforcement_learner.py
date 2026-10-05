from itertools import count

from abc import ABC, abstractmethod

class NStepReinforcementLearner(ABC):
    def __init__(self, mdp, bandit, qfunction, n):
        self.mdp = mdp
        self.bandit = bandit
        self.qfunction = qfunction
        self.n = n

    def execute(self, episodes=100, max_episode_length=float("inf")):
        episode_rewards = []
        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            rewards = []
            states = [state]
            actions = [action]

            episode_reward = 0.0
            done = False
            for step in count():
                # If there are still actions to be tried
                if not done:
                    (next_state, reward, done) = self.mdp.execute(state, action)
                    rewards.append(reward)
                    states.append(next_state)
                    next_actions = self.mdp.get_actions(next_state)
                    next_action = self.bandit.select(
                        next_state, next_actions, self.qfunction
                    )
                    actions.append(next_action)
                    episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                # If we have enough rewards to update the Q-function
                if step >= self.n:
                    n_step_rewards = sum(
                        self.mdp.discount_factor**i * rewards[i]
                        for i in range(min(self.n, len(rewards)))
                    )

                    if (
                        len(states) > self.n
                        and not self.mdp.is_terminal(states[self.n])
                    ):
                        n_step_rewards += (
                            self.mdp.discount_factor**self.n
                            * self.state_value(states[self.n], actions[self.n])
                        )

                    q_value = self.qfunction.get_q_value(states[0], actions[0])

                    self.qfunction.update(
                        states[0],
                        actions[0],
                        n_step_rewards - q_value,
                    )
                    rewards = rewards[1:]
                    states = states[1:]
                    actions = actions[1:]

                # If we have reached maximum epison length, set 'done' to True
                if step == max_episode_length - 1:
                    done = True

                if len(states) == 0:
                    break

                state = next_state
                action = next_action
 
            episode_rewards.append(episode_reward)

        return episode_rewards
        
    """ Get the value of a state """

    @abstractmethod
    def state_value(self, state, action):
        pass
