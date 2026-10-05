from mastering_rl.ppo import PPO
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.value_functions.deep_value_function import DeepValueFunction
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.tests.plot import Plot

gridworld = GridWorld()

# Instantiate the critic
critic = DeepValueFunction(state_space=len(gridworld.get_initial_state()), hidden_dim=64)

from mastering_rl.value_functions.tabular_value_function import TabularValueFunction
critic = TabularValueFunction()

# Instantiate the actor
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
actor = DeepNeuralNetworkPolicy(state_space, action_space)

learner = PPO(gridworld, actor, critic)
rewards = learner.execute(500)

print(gridworld.value_function_to_string(critic))
print(gridworld.stochastic_policy_to_string(actor))

Plot.plot_cumulative_rewards(["PPO"], [rewards], smoothing_factor=0.8)
