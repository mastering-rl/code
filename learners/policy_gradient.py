from itertools import count

from mastering_rl.learners.model_free_learner import ModelFreeLearner


class PolicyGradient(ModelFreeLearner):
    def __init__(self, mdp, policy) -> None:
        super().__init__()
        self.mdp = mdp
        self.policy = policy

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100, max_episode_length=float("inf")):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            episode_reward = 0.0
            for step in count():
                action = self.policy.select_action(state, self.mdp.get_actions(state))
                (next_state, reward, done) = self.mdp.execute(state, action)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length - 1:
                    break

            deltas = self.calculate_deltas(rewards)

            self.policy.update(states, actions, deltas)
            episode_rewards.append(episode_reward)

        return episode_rewards

    def calculate_deltas(self, rewards):
        returns = []
        return_t = 0

        for reward in reversed(rewards):
            return_t = reward + self.mdp.get_discount_factor() * return_t
            returns.insert(0, return_t)

        return returns

