import torch

from gridworld import GridWorld
from experience_replay_learner import ExperienceReplayLearner
from deep_q_function import DeepQFunction
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot


gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())
policy_qfunction = DeepQFunction(state_space, action_space)
target_qfunction = DeepQFunction(state_space, action_space)

learner = ExperienceReplayLearner(gridworld, EpsilonGreedy(), policy_qfunction, target_qfunction, update_period=1)
rewards = learner.execute(episodes=1000)

policy = QPolicy(policy_qfunction)
gridworld.visualise_q_function(policy_qfunction)
gridworld.visualise_policy_as_image(policy)
gridworld.visualise_q_function(target_qfunction)
