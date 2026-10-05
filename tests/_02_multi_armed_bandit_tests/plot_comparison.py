from mastering_rl.multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.multi_armed_bandit.softmax import Softmax
from mastering_rl.multi_armed_bandit.ucb import UpperConfidenceBounds
from mastering_rl.tests.plot import Plot

from mastering_rl.tests._02_multi_armed_bandit_tests.run_bandit import run_bandit


def plot_comparison(drift=False):
    epsilon_greedy = run_bandit(EpsilonGreedy(epsilon=0.1), drift=drift)
    epsilon_decreasing = run_bandit(EpsilonDecreasing(alpha=0.99), drift=drift)
    softmax = run_bandit(Softmax(tau=1.0), drift=drift)
    ucb = run_bandit(UpperConfidenceBounds(), drift=drift)

    Plot.plot_rewards(
        [
            "Epsilon Greedy (epsilon = 0.1)",
            "Epsilon Decreasing (alpha = 0.99)",
            "Softmax (tau = 1.0)",
            "UCB",
        ],
        [epsilon_greedy, epsilon_decreasing, softmax, ucb],
    )


plot_comparison(drift=False)
