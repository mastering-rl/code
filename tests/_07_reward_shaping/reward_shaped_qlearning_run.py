from mastering_rl.learners.qlearning import QLearning
from mastering_rl.learners.reward_shaping.gridworld_bad_potential_function import (
    GridWorldBadPotentialFunction,
)
from mastering_rl.learners.reward_shaping.gridworld_potential_function import (
    GridWorldPotentialFunction,
)
from mastering_rl.learners.reward_shaping.reward_shaped_qlearning import (
    RewardShapedQLearning,
)
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.tests.plot import Plot

print("==========\nTabular Q-learning: Gridworld\n==========")
mdp = GridWorld(width=10, height=7, goals=[((9, 6), 1), ((8, 6), -1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
q_learning_rewards = mdp.get_rewards()

print("==========\nReward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld(width=10, height=7, goals=[((9, 6), 1), ((8, 6), -1)])
qfunction = QTable()
potential = GridWorldPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
shaped_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward shaping"],
    [q_learning_rewards, shaped_rewards],
)

print("==========\nBad Reward Shaped Q-learning: Gridworld\n==========")
mdp = GridWorld()
qfunction = QTable()
potential = GridWorldBadPotentialFunction(mdp)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
bad_shaped_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning 10x7", "Reward shaping 10x7", "Bad reward shaping 4x3"],
    [q_learning_rewards, shaped_rewards, bad_shaped_rewards],
)
