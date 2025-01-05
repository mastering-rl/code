from mastering_rl.learners.n_step_qlearning import NStepQLearning
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.learners.sarsa import SARSA
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.tests.plot import Plot


def qlearning_vs_sarsa(mdp_q, mdp_s, episodes):

    # Train using Q-learning
    qfunction = QTable()
    QLearning(mdp_q, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)

    # Extract and execute the policy and get all rewards: splits into training and test
    policy = StochasticQPolicy(qfunction, EpsilonGreedy(epsilon=0.01))
    mdp_q.execute_policy(policy, episodes=episodes)
    qlearning_rewards = mdp_q.get_rewards()

    # Train using SARSA
    qfunction = QTable()
    SARSA(mdp_s, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)

    # Extract and execute the policy and get all rewards: splits into training and test
    policy = StochasticQPolicy(qfunction, EpsilonGreedy(epsilon=0.01))
    mdp_s.execute_policy(policy, episodes=episodes)
    sarsa_rewards = mdp_s.get_rewards()

    Plot.plot_rewards_per_episode(
        ["Q-learning", "SARSA"],
        [qlearning_rewards, sarsa_rewards],
        smoothing_factor=0.9,
    )


def qlearning_vs_nstep(mdp_q, mdp_n, episodes, n):

    # Train using Q-learning
    qfunction = QTable()
    QLearning(mdp_q, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)

    # Extract and execute the policy and get all rewards: splits into training and test
    policy = StochasticQPolicy(qfunction, EpsilonGreedy(epsilon=0.01))
    mdp_q.execute_policy(policy, episodes=episodes)
    qlearning_rewards = mdp_q.get_rewards()

    # Train using NStep Q-learning
    qfunction = QTable()
    NStepQLearning(mdp_n, EpsilonGreedy(epsilon=0.2), qfunction, n).execute(
        episodes=episodes
    )

    # Extract and execute the policy and get all rewards: splits into training and test
    policy = StochasticQPolicy(qfunction, EpsilonGreedy(epsilon=0.01))
    mdp_n.execute_policy(policy, episodes=episodes)
    nstep_rewards = mdp_n.get_rewards()

    Plot.plot_rewards_per_episode(
        ["Q-learning", "n-step Q-learning"], [qlearning_rewards, nstep_rewards]
    )
