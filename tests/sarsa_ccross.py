from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

from mastering_rl.learners.sarsa import SARSA
from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable

mdp = ContestedCrossing()
qfunction = QTable()
SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

# mdp.visualise_q_function(qfunction)
policy = QPolicy(qfunction)
mdp.visualise_policy(policy, "Policy plot", mode=0)
mdp.visualise_policy(policy, "Path plot", mode=1)
