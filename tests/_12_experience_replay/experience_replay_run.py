from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.tests.plot import Plot


gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())

policy_qfunction = DeepQFunction(state_space, action_space)
target_qfunction = DeepQFunction(state_space, action_space)
#from mastering_rl.qfunctions.qtable import QTable
#policy_qfunction = QTable()
#target_qfunction = QTable()

learner = ExperienceReplayLearner(
    gridworld, EpsilonGreedy(), policy_qfunction, target_qfunction, update_period=1
)
rewards = learner.execute(episodes=200)

policy = QPolicy(policy_qfunction)
gridworld.visualise_q_function(policy_qfunction)
gridworld.visualise_policy_as_image(policy)
