from itertools import count

from mastering_rl.learners.model_free_learner import ModelFreeLearner


class ActorCritic(ModelFreeLearner):
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100, max_episode_length=float("inf")):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            deltas = []

            state = self.mdp.get_initial_state()
            action = self.actor.select_action(state, self.mdp.get_actions(state))
            episode_reward = 0.0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                next_action = self.actor.select_action(
                    next_state, self.mdp.get_actions(next_state)
                )

                delta = self.get_delta(
                    reward, state, action, next_state, next_action, done
                )

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                deltas.append(delta)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length:
                    break

            self.update_critic(states, actions, deltas)
            self.update_actor(states, actions, deltas)
            episode_rewards.append(episode_reward)

        return episode_rewards

    def calculate_deltas(self, states, actions, rewards):
        G = []
        G_t = 0

        for r in reversed(rewards):
            G_t = r + self.mdp.get_discount_factor() * G_t
            G.insert(0, G_t)

        values = [
            self.state_value(state, action) for state, action in zip(states, actions)
        ]
        deltas = [reward - value for reward, value in zip(G, values)]
        return deltas

    def get_delta(self, reward, state, action, next_state, next_action, done):
        q_value = self.state_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = (
            reward
            + (self.mdp.get_discount_factor() * next_state_value * (1 - done))
            - q_value
        )
        return delta

    def update_actor(self, states, actions, deltas):
        abstract

    def update_critic(self, states, actions, deltas):
        abstract
