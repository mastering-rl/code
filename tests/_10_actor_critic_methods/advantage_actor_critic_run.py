from mastering_rl.learners.advantage_actor_critic import AdvantageActorCritic
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.value_functions.deep_value_function import DeepValueFunction
from mastering_rl.markov_decision_processes.gridworld import GridWorld

from mastering_rl.tests.plot import Plot

gridworld = GridWorld()

# Instantiate the actor
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
actor = DeepNeuralNetworkPolicy(state_space, action_space)   

# Instantiate the critic

critic = DeepValueFunction(state_space)

rewards = AdvantageActorCritic(gridworld, actor, critic).execute(300, max_episode_length=300)
gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {1000} iterations")
gridworld.visualise_stochastic_policy(actor)

Plot.plot_cumulative_rewards(["AAC"], [rewards], smoothing_factor=0.8)
