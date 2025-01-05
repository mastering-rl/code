from mastering_rl.learners.qlearning import QLearning
from mastering_rl.learners.sarsa import SARSA
from mastering_rl.markov_decision_processes.gridworld import CliffWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.tests.plot import Plot

## Train using Q-learning

cliffworld = CliffWorld()
cliffworld_image = cliffworld.visualise()

qfunction = QTable()
rewards = QLearning(cliffworld, EpsilonGreedy(epsilon=0.2), qfunction).execute(
    episodes=2000
)

policy = QPolicy(qfunction)
cliffworld.visualise_policy(policy)

## Train using SARSA

cliffworld = CliffWorld()
qfunction = QTable()
rewards = SARSA(cliffworld, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)

policy = QPolicy(qfunction)
cliffworld.visualise_policy(policy)