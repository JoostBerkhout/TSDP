"""
Created on 7/18/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: This script is used to do the numerical experiments for a small
sparse instance to test the performance of the method compared to an LP
approach (which does not work for the larger sparse networks for road,
Moreno and email).
"""

import networkx
from tabulate import tabulate

from TSDP_optimization import *
from utilities import (
    store_results, find_cliques_from_adjacency_matrix,
    calc_stationary_distribution, ergodic_project,
    )

"""
WARNING: 

Choose the following constant values in TSDP_optimization.py:

PRECISION = 10 ** (-5)
BINSEARCH_PRECISION = 10 ** (-5)

"""

# user init
timeLimitLP = 5 * 60
normType = 'inf'
n = 100  # size random instances
numb_exp = 25
verbose = False
epsFR1SH = 10 ** (-8)

# technical init
results_dict = {}

for random_seed in range(numb_exp):

    np.random.seed(random_seed)
    print(f'Start experiment with seed {random_seed}')

    # random preferential attachment network
    G = networkx.generators.random_graphs.barabasi_albert_graph(
        n, 5,
        seed=random_seed
        )
    A = networkx.linalg.graphmatrix.adjacency_matrix(G).todense()
    P = np.array(A / np.sum(A, axis=1))

    mu = calc_stationary_distribution(P)

    # Experiment with more difficult goal
    # # mu_goal where largest clique is uniform
    # mu_goal = mu.copy()
    # cliques = find_cliques_from_adjacency_matrix(P)
    # cliques_length = [len(clq) for clq in cliques]
    # largest_clique_idx = np.argmax(cliques_length)
    # clique_idxs = cliques[largest_clique_idx]
    # sum_max = np.sum(mu_goal[clique_idxs])
    # mu_goal[clique_idxs] = sum_max / len(clique_idxs)

    # Experiment with an easier goal
    # mu_goal where influence of largest clique reduced by 10%
    mu_goal = mu.copy()
    cliques = find_cliques_from_adjacency_matrix(P)
    cliques_length = [len(clq) for clq in cliques]
    largest_clique_idx = np.argmax(cliques_length)
    clique_idxs = cliques[largest_clique_idx]
    clique_mass_increase_factor = .9
    mu_goal[clique_idxs] *= clique_mass_increase_factor
    sum_clique = np.sum(mu_goal[clique_idxs])
    outside_clique = [i for i in range(n) if i not in clique_idxs]
    increase_factor_outside = (1 - sum_clique) / np.sum(mu_goal[outside_clique])
    mu_goal[outside_clique] *= increase_factor_outside

    method_name = 'min_norm_rank_1()'
    start_time = time.time()
    DeltaSol = min_norm_rank_1(P, mu_goal, normType)
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time, calc_norm=True
        )

    # method_name = 'optimal'
    # start_time = time.time()
    # DeltaSol = goal_MC_optimization(
    #         P, mu_goal, normtype=normType,
    #         rankOne=False, ensureNonNeg=True,
    #         verbose=False, timeLimit=timeLimitLP
    # )
    # run_time = time.time() - start_time
    # store_results(
    #         results_dict, method_name, P, DeltaSol, mu_goal, normType,
    #         run_time
    # )

    method_name = 'min_norm_rank_1_pert_pres_stoch()'
    start_time = time.time()
    alpha, soft_mu_goal = max_convex_softened_mu_goal(P, mu, mu_goal)
    if alpha > 0:
        DeltaSol = min_norm_rank_1_pert_pres_stoch(
            P, mu, soft_mu_goal,
            normtype=normType
            )
    else:
        DeltaSol = np.zeros((n, n))
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'R1SH(2)'
    start_time = time.time()
    DeltaSol = RISH_K(
        P, mu, mu_goal, normType=normType,
        verbose=False, sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        numbIntervals=2
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'R1SH(4)'
    start_time = time.time()
    DeltaSol = RISH_K(
        P, mu, mu_goal, normType=normType,
        verbose=False, sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        numbIntervals=4
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'R1SH(8)'
    start_time = time.time()
    DeltaSol = RISH_K(
        P, mu, mu_goal, normType=normType,
        verbose=False, sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        numbIntervals=8
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'R1SH(16)'
    start_time = time.time()
    DeltaSol = RISH_K(
        P, mu, mu_goal, normType=normType,
        verbose=False, sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        numbIntervals=16
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'R1SH'
    start_time = time.time()
    DeltaSol = R1SH(
        P, mu, mu_goal,
        normType=normType,
        verbose=False,
        sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = f'FR1SH({epsFR1SH})'
    start_time = time.time()
    DeltaSol = FR1SH(
        P, mu, mu_goal, normType=normType,
        verbose=True, phi=epsFR1SH,
        sequence='descending',
        jumpToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True
        )
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

    method_name = 'Riesz projector'
    start_time = time.time()
    DeltaSol = DeltaSolErgProj = ergodic_project(mu_goal) - P
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

# print table of results
results_table = [[method] + [np.round(np.nanmean(results[k]), 4)
                             for k, v in results.items()]
                 for method, results in results_dict.items()]
column_labels = ['Method', f'mean(\|Delta\|_{normType})', 'Fraction feasible',
                 'mean run time', 'mean rank']
print(tabulate(results_table, column_labels))
print(tabulate(results_table, column_labels, tablefmt="latex"))
