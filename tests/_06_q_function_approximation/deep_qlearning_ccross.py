from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction


mdp = ContestedCrossing()
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())
qfunction = DeepQFunction(state_space, action_space)
rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=300)
policy = StochasticQPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy_as_image(policy, mode=0,
    title="Low danger: {0}, High danger: {1}".format(mdp.low_danger, mdp.high_danger),
    plot=True
    )
