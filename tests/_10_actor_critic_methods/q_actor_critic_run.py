from mastering_rl.learners.q_actor_critic import QActorCritic
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.qfunctions.qtable import QTable

mdp = GridWorld()
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

# Instantiate the actor
actor = DeepNeuralNetworkPolicy(state_space, action_space)

# Instantiate the critic
critic = QTable()

#  Instantiate the actor critic agent
learner = QActorCritic(mdp, actor, critic)
episode_rewards = learner.execute(episodes=1000)
mdp.visualise_q_function(critic)
mdp.visualise_stochastic_policy(actor)
