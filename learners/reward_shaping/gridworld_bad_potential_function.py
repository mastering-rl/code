from mastering_rl.learners.reward_shaping.gridworld_potential_function import (
    GridWorldPotentialFunction,
)


class GridWorldBadPotentialFunction(GridWorldPotentialFunction):
    def get_potential(self, state):
        return -super().get_potential(state)
