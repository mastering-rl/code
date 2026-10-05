import torch
import matplotlib.pyplot as plt
import cv2

from mastering_rl.learners.advantage_actor_critic import AdvantageActorCritic
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.value_functions.deep_value_function import DeepValueFunction
from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.tests.plot import Plot
from mastering_rl.tests.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


plt.ion()  # Turn on interactive mode for real-time plotting

#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger_21Sept.policy"
#version = "ALE/Breakout-ram-v5"
#policy_name = "Breakout.policy"
#version = 'Frostbite-ramNoFrameskip-v4'
#policy_name = 'Frostbite.policy'
#version = 'ALE/Frostbite-ram-v5'
#version = "Riverraid-ramNoFrameskip-v4"
#version = "Freeway-ramDeterministic-v4"
#policy_name = "Riverraid-v4.policy"
#version = 'CartPole-v1'
#policy_name = "CartPole.policy"
version = "LunarLander-v3"
policy_name = "LunarLander-v3.policy"


mdp = ALEWrapper(version=version)

# Get number of actions and state size
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 3
epochs = 20
epoch_size = 20
train_rewards_all_runs = []
test_rewards_all_runs = []
for r in range(runs):
    # Instantiate the critic
    critic = DeepValueFunction(state_space, hidden_dim=128, alpha=1e-3)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space, hidden_dim=128, alpha=3e-4)

    learner = AdvantageActorCritic(mdp, actor, critic)

    train_rewards, test_rewards = train(
        mdp,
        actor,
        policy_name,
        learner,
        learner_name=f"AAC_{r}",
        test=True,
        max_episode_length=500.0,
        epochs=epochs,
        epoch_size=epoch_size,
    )
    train_rewards_all_runs.append(train_rewards)
    test_rewards_all_runs.append(test_rewards)

labels = ["AAC train " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, train_rewards_all_runs, smoothing_factor=0.5)

test_labels = ["AAC test " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(
    test_labels,
    test_rewards_all_runs,
    smoothing_factor=0.3,
    episodes_per_evaluation=epoch_size,
)

mdp = ALEWrapper(version=version, render_mode="rgb_array")
mdp.create_gif(actor, "assets/gifs/lunar_lander_v3_advantage_actor_critic")