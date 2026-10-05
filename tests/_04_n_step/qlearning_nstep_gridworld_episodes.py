from mastering_rl.markov_decision_processes.contested_crossing import ContestedCrossing
from mastering_rl.markov_decision_processes.gridworld import CliffWorld, GridWorld
from mastering_rl.tests.compare_convergence_curves import qlearning_vs_nstep
from mastering_rl.tests.plot import Plot

mdp_q = GridWorld()
mdp_s = GridWorld()
# mdp_q = ContestedCrossing()
# mdp_s = ContestedCrossing()
qlearning_vs_nstep(mdp_q, mdp_s, episodes=2000, n=3)
