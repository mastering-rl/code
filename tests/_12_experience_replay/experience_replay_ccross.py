import torch

from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.learners.experience_replay_learner import PrioritisedReplayMemory
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.tests.plot import Plot

runs = 5
episodes = 500
all_q_learner_rewards = [] 
all_experience_replay_rewards = []
all_prioritised_experience_replay_rewards = []

for i in range(runs):
    mdp = ContestedCrossing()
    action_space = len(mdp.get_actions())
    state_space = len(mdp.get_initial_state())

    from mastering_rl.qfunctions.dqn import DQN

    #qfunction = DeepQFunction(state_space, action_space)
    qfunction = DQN(state_space, action_space)
    q_learner = QLearning(mdp, EpsilonGreedy(), qfunction)
    q_learner_rewards = q_learner.execute(episodes=episodes)
    all_q_learner_rewards.append(q_learner_rewards)
    print(f'run {i}a')

    mdp = ContestedCrossing()
    policy_qfunction = DQN(state_space, action_space)
    target_qfunction = DQN(state_space, action_space)

    experience_replay_learner = ExperienceReplayLearner(
        mdp, EpsilonGreedy(), policy_qfunction, target_qfunction, update_period=20
    )
    experience_replay_rewards = experience_replay_learner.execute(episodes=episodes)
    all_experience_replay_rewards.append(experience_replay_rewards)
    print(f'run {i}b')
    '''
    mdp = ContestedCrossing()
    policy_qfunction = DeepQFunction(state_space, action_space)
    target_qfunction = DeepQFunction(state_space, action_space)

    prioritised_experience_replay_learner = ExperienceReplayLearner(
        mdp, EpsilonGreedy(), policy_qfunction, target_qfunction, memory=PrioritisedReplayMemory(), update_period=20
    )
    prioritised_experience_replay_rewards = prioritised_experience_replay_learner.execute(episodes=episodes)
    all_prioritised_experience_replay_rewards.append(prioritised_experience_replay_rewards)
    print(f'run {i}c')
    '''
labels = ["Q learning", "Experience replay"]
reward_list = [all_q_learner_rewards, all_experience_replay_rewards]
#labels = ["Q learning", "Experience replay", "Prioritised experience replay"]
#reward_list = [all_q_learner_rewards, all_experience_replay_rewards, all_prioritised_experience_replay_rewards]


Plot.plot_cumulative_rewards_with_variance(labels, reward_list, smoothing_factor=0.95)
"""
policy = QPolicy(policy_qfunction)
mdp.visualise_q_function(policy_qfunction)
mdp.visualise_as_image(
    policy=policy,
    mode=0,
    title="Low danger: {0}, High danger: {1}".format(mdp.low_danger,mdp.high_danger),
    plot=True
)
"""