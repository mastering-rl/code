from mastering_rl.gif_makers.gif_maker import GifMaker
from mastering_rl.learners.policy_gradient import PolicyGradient
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.policies.deep_nn_policy import DeepNeuralNetworkPolicy

gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld)
state_space = len(gridworld.get_initial_state())
action_space = len(gridworld.get_actions())
policy = DeepNeuralNetworkPolicy(state_space=state_space, action_space=action_space)
for episode in range(0, 100):
    title = "Episode %d" % (episode)
    image_texts = gridworld.visualise_stochastic_policy(policy, title=title, gif=True)
    gif_maker.add_frame(image_texts, title=title)
    PolicyGradient(gridworld, policy).execute(episodes=1)

gif_maker.save("assets/gifs/deep_policy_gradient.gif")
