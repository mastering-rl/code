from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.softmax import Softmax
from tests.plot import Plot

from mastering_rl.learners.double_qlearning import DoubleQLearning
from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.learners.reward_shaping.freeway_potential_function import (
    FreewayPotentialFunction,
)
from mastering_rl.learners.reward_shaping.reward_shaped_qlearning import (
    RewardShapedQLearning,
)
from mastering_rl.learners.sarsa import SARSA
from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.markov_decision_processes.freeway import Freeway
from mastering_rl.markov_decision_processes.freeway_abstraction import FreewayAbstraction
from mastering_rl.policies.freeway_hand_policy import FreewayHandPolicy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.qfunctions.qtable import QTable

# version="Breakout-ramDeterministic-v4"
version = "Freeway-ramDeterministic-v4"
# version="ALE/Frogger-ram-v5"
print("==========\nQ-learning: " + version + "\n==========")

# mdp = FreewayAbstraction(render_mode="rgb_array", discount_factor=0.99)
# mdp = Freeway(render_mode="rgb_array", discount_factor=0.99)

mdp = ALEWrapper(version=version, render_mode="rgb_array", discount_factor=1.0)
qfunction = DeepQFunction(
    mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions())
)
qfunction2 = DeepQFunction(
    mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions())
)


# QLearning(mdp, Softmax(), qfunction).execute(episodes=1)
# ExperienceReplayLearner(mdp, EpsilonGreedy(), qfunction, replay_period=100, batch_size=6).execute(episodes=1)
# policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))
# rewards = mdp.execute_policy(policy, episodes=1)
# print("Episode 0: {:.2f}".format(rewards[0]))


episodes = 200
episodes_per_evaluation = 10
epsilon = 1.0
epsilon_decay = 0.995
rewards = []
for i in range(int(episodes / episodes_per_evaluation)):
    rewards += DoubleQLearning(
        mdp, EpsilonGreedy(epsilon=epsilon), qfunction, qfunction2
    ).execute(episodes=episodes_per_evaluation)
    # ExperienceReplayLearner(mdp, EpsilonGreedy(), qfunction).execute(episodes=episodes_per_evaluation)
    # rewards += mdp.execute_policy(policy, episodes=1)
    print(rewards)
    print(
        f"Episode: {i * episodes_per_evaluation}, Epsilon: {epsilon}, Episode Reward: {rewards[-1]}"
    )
    epsilon = max(epsilon * (epsilon_decay**episodes_per_evaluation), 0.01)
    # print("Episode {}: {:.2f}".format((i + 1) * episodes_per_evaluation, rewards[-1-episodes_per_evaluation:-1]))
    qfunction.save("freeway.policy")

qfunction.load("freeway.policy")

"""
mdp = ALEWrapper(version=version, render_mode="rgb_array", discount_factor=0.95)

potential = FreewayPotentialFunction(mdp)

qfunction = DeepQFunction(mdp, state_space=len(mdp.get_initial_state()), action_space=len(mdp.get_actions()), hidden_dim=20)
RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=1)
policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.0))
shaped = mdp.execute_policy(policy, episodes=1)
for i in range(int(episodes / episodes_per_evaluation)):
    print("Episode %d" % (i * episodes_per_evaluation))
    RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes=episodes_per_evaluation)
    shaped += mdp.execute_policy(policy, episodes=1)

Plot.plot_cumulative_rewards(
    ["Q-learning", "Reward-shaped Q-learning"], [rewards, shaped], smoothing_factor=0.9, episodes_per_evaluation=episodes_per_evaluation
)
"""

Plot.plot_cumulative_rewards(
    ["Q-learning"],
    [rewards],
    smoothing_factor=0.0,
    episodes_per_evaluation=episodes_per_evaluation,
)

policy = StochasticQPolicy(qfunction, mdp.get_actions(), EpsilonGreedy(epsilon=0.05))

# policy = FreewayHandPolicy()
# mdp = Freeway(render_mode="human")

# mdp = ALEWrapper(version=version, render_mode="human")
mdp = Freeway(
    version="Freeway-ramDeterministic-v4", render_mode="human", discount_factor=1.0
)
exec_rewards = mdp.execute_policy(policy, episodes=1)
print(exec_rewards)
