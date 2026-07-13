from mastering_rl.learners.value_iteration import ValueIteration
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.stochastic_value_policy import StochasticValuePolicy
from mastering_rl.policies.value_policy import ValuePolicy
from mastering_rl.tests.plot import Plot
from mastering_rl.value_functions.tabular_value_function import TabularValueFunction

maze = GridWorld.open("mastering_rl/layouts/maze.txt")
values = TabularValueFunction()
ValueIteration(maze, values).value_iteration(max_iterations=100)
maze.visualise_value_function(values, grid_size=0.8, title="100 iterations")
policy = ValuePolicy(maze, values)
maze.visualise_policy(policy, "", grid_size=0.8)
