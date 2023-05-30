"""
Created on 7/18/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: This script is used to do the numerical experiments for the
random instances.

WARNING: The following values were used in TSDP_optimization.py for the
experiments

PRECISION = 10 ** (-8)
BINSEARCH_PRECISION = 10 ** (-8)
"""

from tabulate import tabulate

from TSDP_optimization import *
from utilities import (
    store_results, calc_stationary_distribution,
    ergodic_project,
    )

np.random.seed(0)

# user init
timeLimit = 5 * 60
normType = 'inf'
n = 1000  # size random instances
numb_exp = 25
verbose = False

# technical init
results_dict = {}

for random_seed in range(numb_exp):

    np.random.seed(random_seed)
    print(f'Start experiment with seed {random_seed}')

    # random instance
    P = np.random.rand(n, n)
    P = np.diag(1 / np.sum(P, 1)).dot(P)

    mu = calc_stationary_distribution(P)

    # random mu_goal
    mu_goal = np.random.rand(n, 1)
    mu_goal /= sum(mu_goal)
    fraction_random = 0.01
    mu_goal = fraction_random * mu_goal + (1 - fraction_random) * mu

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

    method_name = 'Riesz projector'
    start_time = time.time()
    DeltaSol = DeltaSolErgProj = ergodic_project(mu_goal) - P
    run_time = time.time() - start_time
    store_results(
        results_dict, method_name, P, DeltaSol, mu_goal, normType,
        run_time
        )

# print table of results
results_table = [[method] + [np.round(np.nanmean(results[k]), 6)
                             for k, v in results.items()]
                 for method, results in results_dict.items()]
column_labels = ['Method', f'mean(\|Delta\|_{normType})', 'Fraction feasible',
                 'mean run time', 'mean rank']
print(tabulate(results_table, column_labels))
print(tabulate(results_table, column_labels, tablefmt="latex"))
