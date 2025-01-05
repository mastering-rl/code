from mastering_rl.feature_extractors.gridworld_better_feature_extractor import (
    GridWorldBetterFeatureExtractor,
)
from mastering_rl.feature_extractors.gridworld_feature_extractor import (
    GridWorldFeatureExtractor,
)
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.linear_qfunction import LinearQFunction
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.tests.plot import Plot

mdp = GridWorld()
features = GridWorldFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)


mdp = GridWorld()
features = GridWorldBetterFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
linear_qlearning_rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)


mdp = GridWorld()
qfunction = QTable()
tabular_qlearning_rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute()


Plot.plot_cumulative_rewards(
    ["Tabular Q-learning", "Linear Q-learning"],
    [tabular_qlearning_rewards, linear_qlearning_rewards],
)
