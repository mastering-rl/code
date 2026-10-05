import torch

from mastering_rl.learners.ppo import PPO
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.value_functions.deep_value_function import DeepValueFunction
from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.tests.plot import Plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
version = 'CartPole-v1'
policy_name = "CartPole.policy"

mdp = ALEWrapper(version=version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
episodes = 1000
all_rewards = []
for i in range(runs):

    # Instantiate the critic
    critic = DeepValueFunction(state_space, hidden_dim=128)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space, hidden_dim=128)

    learner = PPO(mdp, actor, critic)
    rewards = learner.execute(episodes)

    all_rewards.append(rewards)

labels = ["PPO " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)
