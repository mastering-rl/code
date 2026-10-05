from mastering_rl.extensive_form_games.backward_induction import BackwardInduction
from mastering_rl.extensive_form_games.tictactoe import TicTacToe
from mastering_rl.utils.graph_visualisation import GraphVisualisation

tictactoe = TicTacToe()
backward_induction = BackwardInduction(tictactoe)
solution = backward_induction.backward_induction(tictactoe.get_initial_state())
gv = GraphVisualisation(max_level=1)
tictactoe_subgraph = gv.node_to_graph(
    tictactoe, solution, print_state=True, print_value=True
)
tictactoe_subgraph.view()

tictactoe = TicTacToe()
backward_induction = BackwardInduction(tictactoe)
state = [["x", "o", "o"], [" ", " ", "x"], [" ", " ", " "]]
next_state = tictactoe.get_transition(state, (1, 1))
solution = backward_induction.backward_induction(next_state)
gv = GraphVisualisation(max_level=100)
tictactoe_subgraph = gv.node_to_graph(
    tictactoe, solution, print_state=True, print_value=True
)
tictactoe_subgraph
