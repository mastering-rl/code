from mastering_rl.gif_makers.gif_maker import GifMaker
from mastering_rl.learners.single_agent_mcts import SingleAgentMCTS
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.ucb import UpperConfidenceBounds
from mastering_rl.qfunctions.qtable import QTable

grid_size = 1.5
gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)
qfunction = QTable()
root_node = None
for time in range(0, 101):
    title = "Time = {:.2f}s".format(time / 100)
    image_texts = gridworld.visualise_q_function(
        qfunction, title=title, grid_size=grid_size, gif=True
    )
    gif_maker.add_frame(image_texts, title=title)
    root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(
        timeout=0.01, root_node=root_node
    )

gif_maker.save("assets/gifs/mcts.gif")
