import torch
import numpy as np
import cv2

from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.learners.q_actor_critic import QActorCritic
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from mastering_rl.tests.train import train
from mastering_rl.tests.plot import Plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)



#version = "ALE/Frogger-ram-v5"
#policy_name = "Frogger_21Sept.policy"
#version = "Breakout-ramNoFrameskip-v4"
#policy_name = "Breakout_22Sept.policy"
#version = 'Frostbite-ramNoFrameskip-v4'
#version = 'ALE/Frostbite-v5'
#version = "Riverraid-ramNoFrameskip-v4"
#version = "Freeway-ramDeterministic-v4"
#policy_name = "Riverraid-v4.policy"
#version = 'CartPole-v1'
#policy_name = "CartPole.policy"
version = "LunarLander-v3"
policy_name = "LunarLander-v3.policy"
#version = "Taxi-v3"
#policy_name = "Taxi-v3.policy"

mdp = ALEWrapper(version=version)

# Get number of actions and state size
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
all_rewards = []
for _ in range(runs):

        # Instantiate the critic
    critic = DeepQFunction(state_space, action_space)

    # Instantiate the actor
    actor = DeepNeuralNetworkPolicy(state_space, action_space)

    learner = QActorCritic(mdp, actor, critic)

    train_rewards, test_rewards = train(
        mdp,
        actor,
        policy_name,
        learner,
        learner_name="",
        test=True,
        plot=True,
        #max_episode_length=500,
        epochs=100,
        epoch_size=20,
    )
    all_rewards.append(test_rewards)

labels = ["Q Actor Critic " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)

# Record 5 evaluation episodes into a single video file.
actor.set_stochastic(False)
all_frames = []
for _ in range(5):
    all_frames.extend(mdp.get_frames(actor, max_episode_length=500))

if all_frames:
    height, width = all_frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        "episode.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    for frame in all_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    print("Saved 5 episodes to episode.mp4")



