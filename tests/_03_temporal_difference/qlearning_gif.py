from mastering_rl.gif_makers.gif_maker import GifMaker
from mastering_rl.learners.qlearning import QLearning
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from mastering_rl.qfunctions.qtable import QTable

grid_size = 1.5
gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)
qfunction = QTable()
for episode in range(1, 101):
    QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=1)
    title = "Episode %d" % (episode)
    image_texts = gridworld.visualise_q_function(
        qfunction, title=title, grid_size=grid_size, gif=True
    )
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("assets/gifs/qlearning.gif")
