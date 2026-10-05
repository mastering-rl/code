from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.tests.plot import Plot


ccross = ContestedCrossing()
action_space = len(ccross.get_actions())
state_space = len(ccross.get_initial_state())

policy_qfunction = DeepQFunction(state_space, action_space)
target_qfunction = DeepQFunction(state_space, action_space)

learner = ExperienceReplayLearner(
    ccross, EpsilonGreedy(), policy_qfunction, target_qfunction, update_period=1
)
rewards = learner.execute(episodes=200)

policy = QPolicy(policy_qfunction)
ccross.visualise_q_function(policy_qfunction)
ccross.visualise_policy_as_image(policy)
