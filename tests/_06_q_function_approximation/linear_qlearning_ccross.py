from mastering_rl.feature_extractors.ccross_feature_extractor import (
    CCrossFeatureExtractor,
)
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes import contested_crossing
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.linear_qfunction import LinearQFunction

mdp = contested_crossing.ContestedCrossing()
features = CCrossFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = StochasticQPolicy(qfunction)
mdp.visualise_as_image(
    policy=policy,
    mode=0,
    title="Low danger: {0}, High danger: {1}".format(mdp.low_danger, mdp.high_danger),
    plot=True,
)
