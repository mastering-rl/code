from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction

gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())
qfunction = DeepQFunction(state_space, action_space)
rewards = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=300)
policy = QPolicy(qfunction)
gridworld.visualise_q_function(qfunction)
gridworld.visualise_policy_as_image(policy)
