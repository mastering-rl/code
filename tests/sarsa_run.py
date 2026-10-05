from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.learners.sarsa import SARSA
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes=10000)
print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))
