from mastering_rl.learners.value_iteration import ValueIteration
from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.stochastic_value_policy import StochasticValuePolicy
from mastering_rl.policies.value_policy import ValuePolicy
from mastering_rl.tests.plot import Plot
from mastering_rl.value_functions.tabular_value_function import TabularValueFunction

## Example: Value iteration for GridWorld

gridworld = GridWorld()
values = TabularValueFunction()

ValueIteration(gridworld, values).value_iteration(max_iterations=100)
gridworld.visualise_value_function(values, "Value function after iteration 100")

policy = ValuePolicy(gridworld, values)
gridworld.visualise_policy(policy, "Policy after iteration 100")


## Evaluating policies

gridworld = GridWorld()
values = TabularValueFunction()
policy = StochasticValuePolicy(gridworld, values)
rewards = gridworld.execute_policy(policy, episodes=1)
for _ in range(50):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    policy = StochasticValuePolicy(gridworld, values)
    rewards += gridworld.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.9)

# Plot the curve on a deterministic version of GridWorld (noise=0.0)
values = TabularValueFunction()
gridworld = GridWorld(noise=0.0)
policy = StochasticValuePolicy(gridworld, values)
rewards = gridworld.execute_policy(policy, episodes=1)
for _ in range(50):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    policy = StochasticValuePolicy(gridworld, values)
    rewards += gridworld.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.0)
Plot.plot_cumulative_rewards(["Value iteration"], [rewards], smoothing_factor=0.9)


## Example: Value iteration for maze solving

maze = GridWorld.open("mastering_rl/layouts/maze.txt")
values = TabularValueFunction()
policy = StochasticValuePolicy(maze, values)
ValueIteration(maze, values).value_iteration(max_iterations=100)
maze.visualise_value_function(values, "Maze value function after iteration 100")
maze.visualise_policy(policy, "Maze policy after iteration 100")


## Example: Value iteration in Contested Crossing

ccross = ContestedCrossing()
values = TabularValueFunction()
ValueIteration(ccross, values).value_iteration(max_iterations=100)
ccross.visualise_value_function(
    values, "Value function after iteration 100", mode=3, cell_size=1.25
)

ccross.visualise_value_function(
    values,
    "Value function after iteration 100, with sub-tables",
    mode=0,
    cell_size=1.25,
)

policy = ValuePolicy(ccross, values)

ccross.visualise_as_image(policy=policy, title="Policy Plot", mode=0, plot=True)
ccross.visualise_as_image(policy=policy, title="Path Plot", mode=1, plot=True)
