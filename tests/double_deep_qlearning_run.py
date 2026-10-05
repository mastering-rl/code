from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.learners.double_qlearning import DoubleQLearning
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy

gridworld = GridWorld()
action_space = len(gridworld.get_actions())
state_space = len(gridworld.get_initial_state())
qfunction1 = DeepQFunction(state_space, action_space)
qfunction2 = DeepQFunction(state_space, action_space)
rewards = DoubleQLearning(gridworld, EpsilonGreedy(), qfunction1, qfunction2).execute(episodes=2000)
policy = QPolicy(qfunction1)
gridworld.visualise_q_function(qfunction1)
gridworld.visualise_policy_as_image(policy)
