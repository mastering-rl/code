from mastering_rl.learners.policy_iteration import PolicyIteration
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.tabular_policy import TabularPolicy

gridworld = GridWorld()
policy = TabularPolicy(default_action=gridworld.LEFT)
PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)
gridworld.visualise_policy(policy)

gridworld = GridWorld(width=20, height=15)
policy = TabularPolicy(default_action=gridworld.LEFT)
iterations = PolicyIteration(gridworld, policy).policy_iteration(max_iterations=100)
print("Number of iterations until convergence: %d" % (iterations))
gridworld.visualise_policy(policy)
