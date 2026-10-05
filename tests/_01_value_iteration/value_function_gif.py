from mastering_rl.gif_makers.gif_maker import GifMaker
from mastering_rl.learners.value_iteration import ValueIteration
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.value_functions.tabular_value_function import TabularValueFunction

grid_size = 1.5
gridworld = GridWorld()
gif_maker = GifMaker(mdp=gridworld, grid_size=grid_size)
values = TabularValueFunction()
for iterations in range(1, 101):
    ValueIteration(gridworld, values).value_iteration(max_iterations=1)
    title = "Iteration %d" % (iterations)
    image_texts = gridworld.visualise_value_function(
        values, title=title, grid_size=grid_size, gif=True
    )
    gif_maker.add_frame(image_texts, title=title)

gif_maker.save("assets/gifs/value_iteration.gif")
