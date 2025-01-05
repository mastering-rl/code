import torch
import numpy as np
import matplotlib.pyplot as plt

from q_actor_critic import QActorCritic
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_q_function import DeepQFunction
from ale_wrapper import ALEWrapper
from tests.plot import Plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


plt.ion()  # Turn on interactive mode for real-time plotting

#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger_21Sept.policy"
#version = "Breakout-ramNoFrameskip-v4"
#policy_name = "Breakout_22Sept.policy"
#version = 'Frostbite-ramNoFrameskip-v4'
#version = 'ALE/Frostbite-v5'
#version = "Riverraid-ramNoFrameskip-v4"
#version = "Freeway-ramDeterministic-v4"
#policy_name = "Riverraid-v4.policy"
version = 'CartPole-v1'
policy_name = "CartPole.policy"
#version = "LunarLander-v2"
#version = "Taxi-v3"

mdp = ALEWrapper(version=version)

# Get number of actions and state size
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
epochs = 200
epoch_size = 10
all_rewards = []
best_reward = float('-inf')
for i in range(runs):

    # Instantiate the critic
    critic = DeepQFunction(state_space, action_space)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space)

    advantage_actor_critic = QActorCritic(mdp, actor, critic)
    train_rewards = []
    test_rewards = []
    for epoch in range(1, epochs + 1):
        train_rewards += advantage_actor_critic.execute(epoch_size, max_episode_length=500)

        test_rewards += mdp.execute_policy(actor, episodes=epoch_size)
        print(
            f"| Epoch: {epoch} | Train reward: {np.mean(train_rewards[-epoch_size:])} | Test reward: {np.mean(test_rewards[-epoch_size:])} |"
        )

        # If this is the best training reward so far, save the policy
        if np.mean(test_rewards[-epoch_size:]) > best_reward:
            actor.save(policy_name)
            best_reward = np.mean(test_rewards[-epoch_size:])
            print(f"Saving {policy_name}")
        
        plt.clf()  # Clear the current figure 
        labels = ["QAC train rewards", "QAC test reward"]
        Plot.plot_cumulative_rewards(labels, [train_rewards, test_rewards], smoothing_factor=0.9)
        plt.draw()
        plt.pause(0.1)

    all_rewards.append(test_rewards)

plt.clf()  # Clear the current figure
plt.ioff() 

mdp = ALEWrapper(version=version, render_mode="human")
actor = DeepNeuralNetworkPolicy.load(state_space, action_space, policy_name)
exec_rewards = mdp.execute_policy(actor, episodes=1)

labels = ["QAC test rewards " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)
