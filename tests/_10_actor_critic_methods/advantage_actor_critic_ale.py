import torch
import matplotlib.pyplot as plt

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
version = 'CartPole-v1'
policy_name = "CartPole.policy"


mdp = ALEWrapper(version=version)

# Get number of actions and state size
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 5
all_rewards = []
for _ in range(runs):
    # Instantiate the critic
    critic = DeepValueFunction(state_space)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space)

    learner = AdvantageActorCritic(mdp, actor, critic)
    train_rewards, test_rewards = train(
        mdp,
        actor,
        policy_name,
        learner,
        learner_name="",
        test=True,
        plot=True,
        max_episode_length=500,
        epochs=200,
        epoch_size=1,
    )
    all_rewards.append(train_rewards)

labels = ["A2C " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)
