from mastering_rl.learners.actor_critic import ActorCritic


class QActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, states, actions, deltas, action_spaces=None):
        self.actor.update(states, actions, deltas, action_spaces=action_spaces)

    def update_critic(self, states, actions, deltas):
        self.critic.batch_update(states, actions, deltas)

    def state_value(self, state, action):
        return self.critic.get_q_value(state, action)