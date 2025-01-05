from mastering_rl.learners.single_agent_mcts import SingleAgentMCTS
from mastering_rl.markov_decision_processes.gridworld import GridWorld
from mastering_rl.multi_armed_bandit.ucb import UpperConfidenceBounds
from mastering_rl.policies.q_policy import QPolicy
from mastering_rl.qfunctions.qtable import QTable
from mastering_rl.utils.graph_visualisation import GraphVisualisation

gridworld = GridWorld()
qfunction = QTable()
root_node = SingleAgentMCTS(gridworld, qfunction, UpperConfidenceBounds()).mcts(
    timeout=0.03
)
gv = GraphVisualisation(max_level=6)
graph = gv.single_agent_mcts_to_graph(root_node, filename="mcts")
graph.view()

gridworld.visualise_q_function(qfunction)

policy = QPolicy(qfunction)
gridworld.visualise_policy(policy)
