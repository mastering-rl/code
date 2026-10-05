import torch

from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.tests.plot import Plot
from mastering_rl.tests.train import train

# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

version = 'CartPole-v1'
policy_name = 'Cartpole.policy'

mdp = ALEWrapper(version, render_mode="rgb_array", discount_factor=0.99)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
epochs = 200
epoch_size = 10
train_rewards_all_runs = []
test_rewards_all_runs = []
for r in range(runs):
    policy_qfunction = DeepQFunction(state_space, action_space, alpha=1e-4)
    target_qfunction = DeepQFunction(state_space, action_space, alpha=1e-4)
    bandit = EpsilonGreedy()
    learner = ExperienceReplayLearner(
        mdp,
        bandit,
        policy_qfunction,
        target_qfunction,
        buffer_size=100000,
        update_period=10,
        min_replay_size=2000,
        target_update_tau=0.01,
    )
    policy = QPolicy(policy_qfunction)

    train_rewards, test_rewards = train(
        mdp,
        policy,
        policy_name,
        learner,
        learner_name=f"Experience_Replay_{r}",
        test=True,
        max_episode_length=500.0,
        epochs=epochs,
        epoch_size=epoch_size,
    )
    train_rewards_all_runs.append(train_rewards)
    test_rewards_all_runs.append(test_rewards)

labels = ["Experience Replay train " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, train_rewards_all_runs, smoothing_factor=0.5)

test_labels = ["Experience Replay greedy eval " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(
    test_labels,
    test_rewards_all_runs,
    smoothing_factor=0.3,
    episodes_per_evaluation=epoch_size,
)

mdp = ALEWrapper(version=version, render_mode="rgb_array")
mdp.create_gif(policy, "assets/gifs/cart_pole_experience_replay")
