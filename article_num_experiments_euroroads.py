"""
Created on 1/16/2022

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: This script is used to do experiments for TSDP_optimization.py
regarding the sparse Euro road network.

WARNING: The following values were used in TSDP_optimization.py for the
experiments

PRECISION = 10 ** (-6)
BINSEARCH_PRECISION = 10 ** (-8)

"""

from tabulate import tabulate

from TSDP_optimization import *
from constants import PRECISION
from utilities import (
    store_results, find_cliques_from_adjacency_matrix,
    calc_stationary_distribution, ergodic_project,
    )

np.random.seed(0)

# user init
normType = 'inf'
verbose = False
numb_exp = 25
epsFR1SH = 10 ** (-8)
clique_mass_increase_factor = 0.9
timeLimit = 60 * 1

# technical init
results_dict = {}

# load email transition matrix
P = np.genfromtxt('Data\\Prepared data\\road_P_scc.csv', delimiter=',')
mu = calc_stationary_distribution(P)
n = len(mu)

# find cliques in order from large to small
cliques = find_cliques_from_adjacency_matrix(P)
cliques_length = np.array([len(clq) for clq in cliques])
cliques_idxs_large_to_small = np.argsort(cliques_length)[::-1]
cliques_to_consider = [cliques[i]
                       for i in cliques_idxs_large_to_small[:numb_exp]]

for idx, clique in enumerate(cliques_to_consider):

    print(f'Start experiment with clique {idx} of length {len(clique)}')

    # mu_goal where largest clique is made less popular
    mu_goal = mu.copy()
    mu_goal[clique] *= clique_mass_increase_factor
    sum_clique = np.sum(mu_goal[clique])
    outside_clique = [i for i in range(n) if i not in clique]
    increase_factor_outside = (1 - sum_clique) / np.sum(mu_goal[outside_clique])
    mu_goal[outside_clique] *= increase_factor_outside
    assert np.abs(np.sum(mu_goal) - 1) <= PRECISION, "Sum of mu_goal is not 1!"

    # check if mu and mu_goal are different
    abs_diff = np.sum(np.abs(mu - mu_goal))
    assert abs_diff > 0, "mu and mu_goal are not different!"

    method_name = 'min_norm_rank_1()'
    start_time = time.time()
    DeltaSol = min_norm_rank_1(P, mu_goal, normType)
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time, calc_norm=True
        )

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
        normType=normType, verbose=True,
        sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True,
        timeLimit=timeLimit
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
        verbose=True, eps=epsFR1SH,
        sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True,
        timeLimit=timeLimit
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
