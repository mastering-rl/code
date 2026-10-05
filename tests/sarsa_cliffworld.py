from mastering_rl.learners.sarsa import SARSA
from mastering_rl.markov_decision_processes.gridworld import CliffWorld
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy


mdp = CliffWorld()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)
print(mdp.q_function_to_string(qfunction))

policy = QPolicy(qfunction)
print(mdp.policy_to_string(policy))
