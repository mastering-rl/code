from mastering_rl.extensive_form_games.sharing_game import SharingGame
from mastering_rl.extensive_form_games.backward_induction import BackwardInduction
from mastering_rl.utils.graph_visualisation import GraphVisualisation


sharing = SharingGame()
backward_induction = BackwardInduction(sharing)
solution = backward_induction.backward_induction(sharing.get_initial_state())

gv = GraphVisualisation(max_level = 5)
graph = gv.node_to_graph(sharing, sharing.game_tree())
graph.view()
