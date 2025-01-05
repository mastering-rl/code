import torch

from deep_nn_policy import DeepNeuralNetworkPolicy
from ale_wrapper import ALEWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
#version = "ALE/Frogger-ram-v5"
#version = 'ALE/Frogger-ram-v4'
#version = 'Frostbite-ramNoFrameskip-v4'
#version = 'ALE/Frostbite-v5'
#version = "Riverraid-ramNoFrameskip-v4"
#version = "Freeway-ramDeterministic-v4"
#policy_name = "Freeway.policy"
#policy_name = "Riverraid-v4.policy"
#version = 'CartPole-v1'
#policy_name = "CartPole.policy"
#policy_name = "Frogger_21Sept.policy"
#version = 'Frostbite-ramNoFrameskip-v4'
#policy_name = 'Frostbite.policy'
version = "Breakout-ramNoFrameskip-v4"
policy_name = "Breakout_22Sept.policy"

mdp = ALEWrapper(version=version, render_mode="human")

# Get number of actions and state size
action_space = len(mdp.get_actions())
state_space = len(mdp.get_initial_state())

actor = DeepNeuralNetworkPolicy.load(state_space, action_space, policy_name)
actor.set_stochastic(False)
exec_rewards = mdp.execute_policy(actor, episodes=1)
