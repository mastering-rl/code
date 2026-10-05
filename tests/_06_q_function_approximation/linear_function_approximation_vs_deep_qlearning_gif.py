from mastering_rl.feature_extractors.gridworld_better_feature_extractor import (
    GridWorldBetterFeatureExtractor,
)
from mastering_rl.gif_makers.side_by_side_comparison import join_gif, run_learner
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.qfunctions.linear_qfunction import LinearQFunction

episodes = 50
gridworld = GridWorld()
features = GridWorldBetterFeatureExtractor(gridworld)
qfunction = LinearQFunction(features)
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="Linear Q-learning",
    out_filename="assets/gifs/linear_qlearning.gif",
    episodes=episodes,
)

gridworld = GridWorld()
qfunction = DeepQFunction(
    state_space=len(gridworld.get_initial_state()),
    action_space=len(gridworld.get_actions()),
    hidden_dim=16,
)
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(
    mdp=gridworld,
    learner=learner,
    qfunction=qfunction,
    learner_name="Deep Q-learning",
    out_filename="assets/gifs/deep_qlearning.gif",
    episodes=episodes,
)


join_gif(
    filename1="assets/gifs/linear_qlearning.gif",
    filename2="assets/gifs/deep_qlearning.gif",
    out_filename="assets/gifs/linear_qlearning_vs_deep_qlearning.gif",
)
