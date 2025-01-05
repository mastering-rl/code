from mastering_rl.learners.sarsa import SARSA
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable

gridworld = GridWorld()
qfunction = QTable()
SARSA(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)