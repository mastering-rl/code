import torch

from mastering_rl.learners.experience_replay_learner import ExperienceReplayLearner
from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from mastering_rl.tests.plot import Plot

# if GPU is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

version = 'CartPole-v1'
policy_name = 'Cartpole.policy'

mdp = ALEWrapper(version, render_mode="rgb_array")

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 2
episodes = 1000
eval_interval = 10
train_rewards_all_runs = []
eval_rewards_all_runs = []
for r in range(runs):
    policy_qfunction = DeepQFunction(state_space, action_space, alpha=1e-4)
    target_qfunction = DeepQFunction(state_space, action_space, alpha=1e-4)
    bandit = EpsilonDecreasing(epsilon=1.0, alpha=0.9995, lower_bound=0.01)
    learner = ExperienceReplayLearner(
        mdp,
        bandit,
        policy_qfunction,
        target_qfunction,
        memory_size=100000,
        update_period=10,
        min_replay_size=2000,
        target_update_tau=0.005,
    )

    train_rewards = []
    eval_rewards = []
    for _ in range(episodes // eval_interval):
        train_rewards += learner.execute(episodes=eval_interval, max_episode_length=500.0)

        greedy_policy = QPolicy(policy_qfunction)
        greedy_eval = mdp.execute_policy(
            greedy_policy,
            episodes=3,
            max_episode_length=500.0,
        )
        eval_rewards.append(sum(greedy_eval) / len(greedy_eval))
        episodes_done = (_ + 1) * eval_interval
        print(f'run {r} | ep {episodes_done}/{episodes} | greedy avg: {eval_rewards[-1]:.1f}')

    train_rewards_all_runs.append(train_rewards)
    eval_rewards_all_runs.append(eval_rewards)
    print(f'run {r}')

labels = ["Experience Replay train " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, train_rewards_all_runs, smoothing_factor=0.5)

eval_labels = ["Experience Replay greedy eval " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(
    eval_labels,
    eval_rewards_all_runs,
    smoothing_factor=0.3,
    episodes_per_evaluation=eval_interval,
)

policy = QPolicy(policy_qfunction)

#mdp = ALEWrapper(version=version, render_mode="rgb_array")
#mdp.create_gif(policy, "cart_pole_experience_replay")
#mdp.create_gif(policy, "../assets/gifs/freeway_trained_deep_q_function_precalculated")
