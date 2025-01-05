from gridworld import GridWorld
from value_iteration import ValueIteration
from value_policy import ValuePolicy
from stochastic_value_policy import StochasticValuePolicy
from tabular_value_function import TabularValueFunction
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

maze = GridWorld.open("../mastering_rl/layouts/maze.txt")
values = TabularValueFunction()
ValueIteration(maze, values).value_iteration(max_iterations=100)
maze.visualise_value_function(values, grid_size=0.8, title="100 iterations")
policy = ValuePolicy(maze, values)
maze.visualise_policy(policy, "", grid_size=0.8)
