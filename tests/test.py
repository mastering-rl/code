from advantage_actor_critic import AdvantageActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from tabular_value_function import TabularValueFunction
from tabular_value_function import TabularValueFunction
from gridworld import GridWorld
from plot import Plot

gridworld = GridWorld()

# Instantiate the actor
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
actor = DeepNeuralNetworkPolicy(state_space, action_space)

# Instantiate the critic
critic = TabularValueFunction()
from deep_value_function import DeepValueFunction

critic = DeepValueFunction(state_space)

aac = AdvantageActorCritic(mdp=gridworld, actor=actor, critic=critic)
rewards = aac.execute(1000)
gridworld.visualise_value_function(critic, grid_size=0.8, title=f"Value Function: {1000} iterations")
gridworld.visualise_stochastic_policy(actor)

Plot.plot_cumulative_rewards(["A2C"], [rewards], smoothing_factor=0.8)
