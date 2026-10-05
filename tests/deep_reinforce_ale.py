import torch
import cv2

from mastering_rl.markov_decision_processes.ale_wrapper import ALEWrapper
from mastering_rl.learners.reinforce import REINFORCE
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy
from mastering_rl.policies.stochastic_q_policy import StochasticQPolicy
from mastering_rl.qfunctions.deep_q_function import DeepQFunction
from mastering_rl.multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from mastering_rl.tests.train import train
from mastering_rl.tests.plot import Plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
# version = "ALE/Frogger-ram-v5"
version = "CartPole-v1"
policy_name = "CartPole.policy"
#version = 'ALE/Frostbite-ram-v5'
#policy_name = "Frostbite.policy"

#version = "LunarLander-v3"
#policy_name = "LunarLander-v3.policy"

#version = "LunarLander-v3"
#policy_name = "LunarLander-v3.policy"

mdp = ALEWrapper(version)

# Get number of actions and state size from gym action space
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

runs = 1
all_rewards = []
for _ in range(runs):
    policy = DeepNeuralNetworkPolicy(state_space, action_space)
    learner = REINFORCE(mdp, policy)
    train_rewards, test_rewards = train(
        mdp,
        policy,
        policy_name,
        learner,
        learner_name="",
        test=True,
        plot=True,
        max_episode_length=500,
        epochs=50,
        epoch_size=20,
    )
    all_rewards.append(test_rewards)

labels = ["REINFORCE " + str(i) for i in range(runs)]
Plot.plot_cumulative_rewards(labels, all_rewards, smoothing_factor=0.9)


# Record 5 evaluation episodes into a single video file.
policy.set_stochastic(False)
all_frames = []
for _ in range(5):
    all_frames.extend(mdp.get_frames(policy, max_episode_length=500))

if all_frames:
    height, width = all_frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        "episode.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    for frame in all_frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video_writer.release()
    print("Saved 5 episodes to episode.mp4")
