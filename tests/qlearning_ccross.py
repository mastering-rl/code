from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.qtable import QTable

print("==========\nTabular Q-learning: Contested crossing\n==========")


episodes = 2000
episodes_per_evaluation = 20
qfunction = QTable()
mdp = ContestedCrossing()
policy = StochasticQPolicy(qfunction, EpsilonGreedy())
rewards = mdp.execute_policy(policy, episodes=1)
for _ in range(int(episodes / episodes_per_evaluation)):
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    policy = StochasticQPolicy(qfunction, EpsilonGreedy())
    rewards += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(
    ["Q-learning"],
    [rewards],
    smoothing_factor=0.0,
    episodes_per_evaluation=episodes_per_evaluation,
)
Plot.plot_cumulative_rewards(
    ["Q-learning"], [rewards], episodes_per_evaluation=episodes_per_evaluation
)
