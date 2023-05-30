"""
Created on 11/22/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Article experiments for Example 2 and Appendix B.2 are done here.

WARNING: The following values were used in TSDP_optimization.py for the
experiments

PRECISION = 10 ** (-8)
BINSEARCH_PRECISION = 10 ** (-8)

"""

import tabulate as TAB

from MC_generators import MMsK_queue
from TSDP_optimization import *
# global init
from utilities import (
    calc_stationary_distribution, check_feasibility_solution,
    calculate_norm, determine_z, determine_u, determine_l, ergodic_project,
    )

normType = 'inf'
numb_decimals = 3
verbose = False

# Example 2
# =========

print('\nFor Example 2:\n')

# instance init
s = 2
K = 1
arr_rate = 1
ser_rate = 1.8
rates_Delta_instances = [(1, .2), (1, 1.2), (1, 1.4), (1, 1.6), (1, 2)]

# technical init
numb_states = s + K + 1
G = MMsK_queue(s, K, arr_rate, ser_rate)
mu = calc_stationary_distribution(G)
results = []
results_appendix_B_2 = []

for arr_rate_Delta, ser_rate_Delta in rates_Delta_instances:

    print(
        f'Start experiments with lambda = {arr_rate_Delta} '
        f'and nu = {ser_rate_Delta}:'
        )

    # calculate goal
    G_Delta = MMsK_queue(s, K, arr_rate_Delta, ser_rate_Delta)
    mu_Delta = calc_stationary_distribution(G_Delta)

    # calculate measures for goal
    z, z_plus, z_min = determine_z(G, mu_Delta)
    l = determine_l(G, mu_Delta)
    u = determine_u(G, mu_Delta)

    # calculate results
    mu_Delta_u = mu.T.dot(u)[0][0]
    d_u = (mu_Delta - mu).T.dot(u)[0][0]
    alpha_star = mu_Delta_u / (1 - d_u)
    alpha_star_2 = max_feas_stepsize_pres_stochasticity(G, mu, mu_Delta)
    alpha, soft_mu_Delta = max_convex_softened_mu_goal(G, mu, mu_Delta)

    # consider Delta's that respect stochasticity
    if alpha == 1:
        # allowed to change support
        Delta = min_norm_rank_1_pert_pres_stoch(
            G,
            mu,
            mu_Delta,
            normtype=normType
            )
        norm_Delta_stoch_rank_1 = calculate_norm(Delta, normType)
    else:
        norm_Delta_stoch_rank_1 = 'no candidate'

    # consider "a" Delta as specified in Theorem 2
    if alpha == 1:
        # allowed to change support
        a_Delta = u.dot(z.T) / (mu_Delta.T.dot(u)[0][0])
        norm_a_Delta = calculate_norm(a_Delta, normType)
    else:
        norm_a_Delta = 'no candidate'

    # no stochasticity
    Delta_no_stoch = min_norm_rank_1(G, mu_Delta, normtype=normType)
    norm_Delta_no_stoch = calculate_norm(Delta_no_stoch, normType)

    # minimum general rank (with stochasticity)
    Delta_gen_rank = goal_MC_optimization(
        G,
        mu_Delta,
        normtype=normType,
        rankOne=False,
        ensureNonNeg=True,
        onlyRowIdxs=None,
        verbose=False,
        DeltaHist=None,
        sameSuppPrecision=None
        )
    if check_feasibility_solution(G, Delta_gen_rank, mu_Delta):
        norm_Delta_stoch_gen_rank = calculate_norm(Delta_gen_rank, normType)
    else:
        norm_Delta_stoch_gen_rank = 'no candidate'

    # R1SH: rank-1 steps heuristic
    R1SH_Delta = R1SH(
        G,
        mu,
        mu_Delta,
        normType=normType,
        verbose=verbose,
        sequence='descending',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True
        )
    if check_feasibility_solution(G, R1SH_Delta, mu_Delta):
        norm_R1SH_Delta = calculate_norm(R1SH_Delta, normType)
    else:
        norm_R1SH_Delta = 'no candidate'

    # finer_R1SH: rank-1 steps heuristic
    finer_R1SH_Delta = FR1SH(
        G,
        mu,
        mu_Delta,
        normType=normType,
        verbose=verbose,
        sequence='descending',
        jumpToGoalIfPossible=False,
        considerDeltaHistory=True,
        trackBestSolutionAlongTheWay=True
        )
    if check_feasibility_solution(G, finer_R1SH_Delta, mu_Delta):
        norm_finer_R1SH_Delta = calculate_norm(finer_R1SH_Delta, normType)
    else:
        norm_finer_R1SH_Delta = 'no candidate'

    # intervals_2_Delta
    intervals_2_Delta = RISH_K(
        G,
        mu,
        mu_Delta,
        normType=normType,
        verbose=verbose,
        sequence='random',
        jumpDirectlyToGoalIfPossible=False,
        considerDeltaHistory=True,
        numbIntervals=2
        )
    if check_feasibility_solution(G, intervals_2_Delta, mu_Delta):
        norm_intervals_2_Delta = calculate_norm(intervals_2_Delta, normType)
    else:
        norm_intervals_2_Delta = 'no candidate'

    # Riesz projector
    Riesz_Delta = ergodic_project(mu_Delta) - G
    norm_Riesz_Delta = calculate_norm(Riesz_Delta, normType)

    # save results
    results.append(
        [
            (arr_rate_Delta, ser_rate_Delta),
            tuple(mu_Delta.flatten().round(3)),
            alpha_star,
            norm_a_Delta,
            norm_Delta_no_stoch,
            norm_Delta_stoch_rank_1,
            ]
        )

    # save results for Appendix B.2 so that we do not have to repeat everything
    results_appendix_B_2.append(
        [
            (arr_rate_Delta, ser_rate_Delta),
            norm_Delta_no_stoch,
            norm_Delta_stoch_gen_rank,
            norm_Delta_stoch_rank_1,
            norm_R1SH_Delta,
            norm_finer_R1SH_Delta,
            norm_intervals_2_Delta,
            norm_Riesz_Delta,
            ]
        )

# round floats in results to numb_decimals
results = [[np.round(r, numb_decimals) if isinstance(r, float) else r
            for r in res] for res in results]

# print latex table
column_labels = [
    u'$\\lambda_\\Delta,\\nu_\\Delta$',
    u'$\mu_\Delta$',
    u'$\\alpha^\\star$',
    u'$\|Delta\|_\infty$ for \eqref{eq:a_rank_1_stoch_solution}',
    u'$\|Delta\|_\infty$',
    u'$\|Delta\|_\infty$ (no stoch.)',
    ]
TAB.LATEX_ESCAPE_RULES = {}
print(TAB.tabulate(results, column_labels))
print(TAB.tabulate(results, column_labels, tablefmt="latex"))

# Appendix B.2
# ============

print('\nFor Appendix B.2:\n')

# round floats in results to numb_decimals
results_appendix_B_2 = [[np.round(r, numb_decimals)
                         if isinstance(r, float) else r
                         for r in res] for res in results_appendix_B_2]

# print latex table
column_labels = [
    u'$\\lambda_\\Delta,\\nu_\\Delta$',
    u'$\min \|Delta\|_\infty$',
    u'$\min_{\Delta \in {\\bf \Delta }^{ \eqref{Gdelta}} \cap {\\bf \Delta }^{\geq 0}} \| \Delta \|$',
    u'$\min_{\Delta \in {{\\bf \Delta }( G, \mu_\Delta )}} \| \Delta\|$',
    u'R1SH',
    u'FR1SH($10^{-3}$)',
    u'R1SH(2)',
    u'Riesz projector',
    ]
TAB.LATEX_ESCAPE_RULES = {}
print(TAB.tabulate(results_appendix_B_2, column_labels, tablefmt="latex"))
print(TAB.tabulate(results_appendix_B_2, column_labels))
