from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from reward_tuned_qlearning import RewardTunedQLearning
from gridworld_reward_tuner import GridWorldRewardTuner
from gridworld_non_potential_function import GridWorldNonPotentialFunction
from q_policy import QPolicy
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.softmax import Softmax
from plot import Plot

print("==========\nTabular Q-learning: Gridworld\n==========")
mdp = GridWorld(width = 10, height = 7, goals = [((9,6), 1), ((8,6), -1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
q_learning_rewards = mdp.get_rewards()

print("==========\nReward Tuned Q-learning: Gridworld\n==========")
mdp = GridWorld(width = 10, height = 7, goals = [((9,6), 1), ((8,6), -1)])
qfunction = QTable()
tuner = GridWorldRewardTuner(mdp)
RewardTunedQLearning(mdp, EpsilonGreedy(), tuner, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
tuned_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning", "Reward tuning"],
    [q_learning_rewards, tuned_rewards],
)

print("==========\nBad Reward Tuned Q-learning: Gridworld\n==========")
mdp = GridWorld(width = 10, height = 7, goals = [((9,6), 1), ((8,6), -1)])
qfunction = QTable()
tuner = GridWorldNonPotentialFunction(mdp)
RewardTunedQLearning(mdp, EpsilonGreedy(), tuner, qfunction).execute(episodes=100)
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)
bad_tuned_rewards = mdp.get_rewards()

Plot.plot_episode_length(
    ["Tabular Q-learning", "Bad Reward tuning"],
    [q_learning_rewards, bad_tuned_rewards],
)