from advantage_actor_critic import AdvantageActorCritic
from mastering_rl.learners.reinforce import REINFORCE
from deep_nn_policy import DeepNeuralNetworkPolicy
from deep_value_function import DeepValueFunction
from tabular_value_function import TabularValueFunction
from gridworld import GridWorld
from contested_crossing import ContestedCrossing
from plot import Plot

#mdp = GridWorld()
mdp = ContestedCrossing()

# Instantiate the critic


# Instantiate the actor
state_space = len(mdp.get_initial_state())
action_space = len(mdp.get_actions())

episodes = 200
runs = 10

reinforce_rewards = []
aac_rewards = []
for run in range(runs):
    actor = DeepNeuralNetworkPolicy(state_space, action_space, hidden_dim=64)
    critic = DeepValueFunction(state_space=state_space, hidden_dim=64)
    policy = DeepNeuralNetworkPolicy(state_space, action_space, hidden_dim=64)

    aac = AdvantageActorCritic(mdp, actor, critic)
    aac_rewards.append(aac.execute(episodes))

    reinforce = REINFORCE(mdp, policy)
    reinforce_rewards.append(reinforce.execute(episodes))


Plot.plot_cumulative_rewards2(["REINFORCE", "A2C"], [reinforce_rewards, aac_rewards])
