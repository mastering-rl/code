from mastering_rl.gif_makers.side_by_side_comparison import join_gif, run_learner
from mastering_rl.learners.n_step_qlearning import NStepQLearning
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.qfunctions.qtable import QTable

episodes = 20
gridworld = GridWorld()
qfunction = QTable()
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="1-Step Q-Learning",
    out_filename="assets/gifs/1_step_qlearning.gif",
    episodes=episodes,
)

gridworld = GridWorld()
qfunction = QTable()
learner = NStepQLearning(gridworld, EpsilonGreedy(), qfunction, 5)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="5-Step Q-Learning",
    out_filename="assets/gifs/5_step_qlearning.gif",
    episodes=episodes,
)

join_gif(
    filename1="assets/gifs/1_step_qlearning.gif",
    filename2="assets/gifs/5_step_qlearning.gif",
    out_filename="assets/gifs/1_step_vs_5_step_qlearning.gif",
)
