from itertools import count

from mastering_rl.learners.model_free_learner import ModelFreeLearner


class REINFORCE(ModelFreeLearner):
    def __init__(self, mdp, policy) -> None:
        super().__init__()
        self.mdp = mdp
        self.policy = policy

    """ Generate and store an entire episode trajectory to use to update the policy """

    def execute(self, episodes=100, max_episode_length=100):
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

    def reset(self):
        self.policy.reset()

    def calculate_deltas(self, rewards):
        G = []
        G_t = 0

        for r in reversed(rewards):
            G_t = r + self.mdp.get_discount_factor() * G_t
            G.insert(0, G_t)

        return G
