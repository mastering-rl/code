from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.tests.plot import Plot


gridworld = GridWorld()
qfunction = QTable()
QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function(qfunction, "Q-Function", grid_size=1.5)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)

qfunction = QTable()
mdp = GridWorld()
rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=2000)
Plot.plot_cumulative_rewards(["Q-learning"], [rewards])

qfunction = QTable()
mdp = ContestedCrossing()
rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=5000)
Plot.plot_cumulative_rewards(["Q-learning (Contested Crossing)"], [rewards])

policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_as_image(
    policy=policy,
    mode=0,
    title="Low danger: {0}, High danger: {1}".format(mdp.low_danger,mdp.high_danger),
    plot=True
)
