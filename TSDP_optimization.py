"""
Created on 3/17/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Module with functions to perform TSDP optimization. Notation from
the article is used as much as possible.
"""

import math
import time

import numpy as np
import pulp

from constants import PRECISION, BINSEARCH_PRECISION, MAX_BINSEARCH_ITER
from utilities import (
    check_feasibility_solution, calculate_norm, determine_d,
    determine_z, determine_u, determine_l,
    )


def find_gamma_for_x_iter_increase(
    idx, list_rates, list_rates_thresholds,
    list_threshold_deltas, x_iter, increase
    ):
    # gamma is the increase of v[idx] if x_iter[idx] increases with 'increase'

    # for notational convenience, get the info for index
    rates = list_rates[idx]
    thresholds = list_rates_thresholds[idx]
    threshold_deltas = list_threshold_deltas[idx]
    new_x_iter_val = x_iter[idx, 0] + increase

    # find gamma
    last_idx_full = np.sum(
        thresholds < new_x_iter_val + PRECISION
        ) - 1  # index of largest threshold that is still < new_x_iter_val
    if last_idx_full >= 0:
        gamma = np.dot(
            threshold_deltas[:last_idx_full + 1],
            rates[:last_idx_full + 1]
            )
        gamma += (new_x_iter_val - thresholds[last_idx_full]) * rates[
            last_idx_full + 1]
    else:
        gamma = increase * rates[last_idx_full + 1]

    return gamma


def find_x_iter_increase_for_gamma(
    idx, list_rates, list_rates_thresholds,
    list_threshold_deltas, x_iter, gamma
    ):
    # finds x_iter_increase that lets v[idx] increase by gamma

    # get the index info
    old_rates = np.array(list_rates[idx])
    thresholds = np.array(list_rates_thresholds[idx])
    threshold_deltas = np.array(list_threshold_deltas[idx])
    x_iter_val = np.array(x_iter[idx, 0])

    # hack so that we can deal with negative areas and positive areas in
    # the same way: it is assumed that the gamma always corresponds to
    # (or "covers") only positive rectangles, or negative rectangles
    # (a rectangle [rect] is the area under the step-wise rates graph)
    gamma = np.abs(gamma)
    rates = np.abs(old_rates)

    # determine x_iter_increase
    rect_sizes = threshold_deltas * rates
    cumsum_rect_sizes = np.cumsum(rect_sizes)
    last_idx_full = np.sum(cumsum_rect_sizes <= gamma) - 1

    # process fully covered rectangles
    if last_idx_full >= 0:
        x_iter_increase = thresholds[
                              last_idx_full
                          ] - x_iter_val  # of fully covered rectangles
        remaining_gamma = gamma - cumsum_rect_sizes[last_idx_full]
    else:
        x_iter_increase = 0  # no fully covered rectangles
        remaining_gamma = gamma

    # process last partly covered rectangle
    last_x_iter_increase = remaining_gamma / rates[last_idx_full + 1]
    x_iter_increase += last_x_iter_increase  # of the partly covered rectangle

    # determine updated (upd) rates, thresholds, and threshold_deltas
    # (useful when gamma increase is processed via x_iter_increase)
    if last_idx_full >= 0:
        upd_rates = old_rates[last_idx_full + 1:]
        upd_thresholds = thresholds[last_idx_full + 1:]
        upd_threshold_deltas = threshold_deltas[last_idx_full + 1:]
    else:
        upd_rates = old_rates
        upd_thresholds = thresholds
        upd_threshold_deltas = threshold_deltas
    upd_threshold_deltas[0] -= last_x_iter_increase

    return x_iter_increase, upd_rates, upd_thresholds, upd_threshold_deltas


def det_gammas_decreases(v_LB, v_iter, cut):
    """Function needed in Phase 1 of
    min_norm_rank_1_pert_pres_stoch_consider_hist() that finds the gamma
    decreases so that v_iter is made equal to cut. If v_iter is already smaller
    than cut, it is ignored. If cut is smaller than v_LB, then v_iter will be
    reduced to v_LB (below v_LB the rate becomes positive).
    """

    # init
    gammas = np.zeros((len(v_iter), 1))

    # determine scenarios
    fall_above = cut > v_iter  # corresponding gammas remain zero
    fall_below = cut < v_LB
    fall_in = np.logical_not(np.logical_or(fall_above, fall_below))

    # set gamma values
    gammas[fall_below] = v_iter[fall_below] - v_LB[fall_below]
    gammas[fall_in] = v_iter[fall_in] - cut

    return gammas


def det_gammas_increase(v_iter, v_UB, cut):
    """Function needed in Phase 2 of
    min_norm_rank_1_pert_pres_stoch_consider_hist() that finds the gamma
    increases so that v_iter is made equal to cut. If v_iter is already larger
    than cut, it is ignored. If cut is larger than v_LB, then v_iter will be
    reduced to v_UB (below v_LB the rate becomes positive).
    """

    # init
    gammas = np.zeros((len(v_iter), 1))

    # determine scenarios
    fall_above = cut > v_UB
    fall_below = cut < v_iter  # corresponding gammas remain zero
    fall_in = np.logical_not(np.logical_or(fall_above, fall_below))

    # set gamma values
    gammas[fall_above] = v_UB[fall_above] - v_iter[fall_above]
    gammas[fall_in] = cut - v_iter[fall_in]

    return gammas


def min_norm_rank_1_pert_pres_stoch_consider_hist(
    G, mu, mu_goal, normtype='inf', u=None, DeltaHist=None
    ):
    """Under that assumption that

        mu_goal.T u >= 1

    it finds an explicit solution to:

        min || DeltaHist + Delta ||_inf
        s.t.
            mu_goal^T (P + Delta) = mu_goal^T
            P + Delta >= 0,

    i.e., Problem (8.3) with the inf-norm. A similar approach is used as the
    algorithms in Appendix D, with the difference that now also the "history of
    previous perturbations" DeltaHist is taken into account. A Binary search
    approach is used to speed up the computation.

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Probability vector of the stationary distribution goal.
    normtype : str
        String indication the norm type to apply. Options: 'inf'.
    u : np.array
        The u vector from the article, see determine_u(). This can be
        given so that it can be reused for speed reasons.
    DeltaHist : np.array
        Sum of historical Delta's.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    if DeltaHist is None:
        return min_norm_rank_1_pert_pres_stoch(
            G, mu, mu_goal, normtype=normtype, u=u
            )

    if u is None:
        u = determine_u(G, mu_goal)

    if mu_goal.T.dot(u) < 1 - PRECISION:
        raise Exception(
            'This function only works when mu_goal.T.dot(u) >= 1,'
            f' its value is now {mu_goal.T.dot(u)}'
            )
    elif mu_goal.T.dot(u) == 1:
        z, z_plus, z_min = determine_z(G, mu_goal)
        DeltaSol = u.dot(z.T) / (mu_goal.T.dot(u))
        return DeltaSol
    elif normtype != 'inf':
        raise NotImplementedError

    # start construction of an optimal solution for inf-norm

    # init
    n = len(G)
    z, z_plus, z_min = determine_z(G, mu_goal)
    abs_z = np.abs(z)
    z_neg = z < 0
    z_pos = z > 0

    """
    Notation used in the remaining function:
        - x_iter: the current solution x
        - v_iter: the current values of the rows in the objective function, 
                  also denoted as v.
        - gammas: the (TBD) decreases of v_iter in each iteration
    """

    """The following is a hack to solve the false assumption that x >= 0, which 
    should be x >= l if we take history into account. The following hack allows
    to use a similar format as min_norm_rank_1_pert_pres_stoch().
    The x_iter can now be thought as the extra mass that is given to l, i.e., 
    "true x solution" = l + x_iter.  
    """
    l = determine_l(G, mu_goal)
    DeltaHist = DeltaHist + l.dot(z.T)
    u = u - l
    goal_value = 1 - mu_goal.T.dot(l)[0][0]  # mu_goal.T.dot(x)'s goal value

    # technical init
    x_iter = np.zeros((n, 1))  # we start increasing x_iter from zero
    val_mu_goal_T_x_iter = 0
    v_iter = np.sum(
        np.abs(DeltaHist),
        axis=1,
        keepdims=True
        )  # v_iter[i] is the current value of the absolute sum of the i-th row of the objective
    sum_abs_z = np.sum(abs_z)
    list_rates = []  # list_rates[i] are the current and upcoming rates of x_iter[i]
    list_rates_thresholds = []  # list_rates_thresholds[i] are the new-rate-thresholds of x_iter[i]
    list_threshold_deltas = []  # list_threshold_deltas[i] are the threshold increments of x_iter[i]

    # determine the rates of increase and thresholds
    for i in range(n):

        # determine the idxs that can lower their corresp. absolute value
        Delta_row_T = DeltaHist[[i], :].T
        diff_signs = np.logical_or(
            np.logical_and(Delta_row_T < 0, z_pos),
            np.logical_and(Delta_row_T > 0, z_neg)
            ).flatten()
        diff_sign_idxs = np.argwhere(diff_signs).flatten()

        if len(diff_sign_idxs) == 0:
            # there is only one rate, no new rates
            list_rates_thresholds.append(np.array([np.inf]))
            list_rates.append(np.array([sum_abs_z]))
            list_threshold_deltas.append(np.array([np.inf]))
            continue

        # determine thresholds for new rates
        new_rates_thresholds = - Delta_row_T[diff_sign_idxs,
                                 :] / z[diff_sign_idxs, :]
        new_rates_thresholds = new_rates_thresholds.flatten()
        argsort_new_rate_threshold = np.argsort(new_rates_thresholds)
        new_rates_thresholds = new_rates_thresholds[argsort_new_rate_threshold]
        new_rates_thresholds = np.append(
            new_rates_thresholds, np.inf
            )  # to ensure we keep increasing with the last current rate
        list_rates_thresholds.append(new_rates_thresholds)

        # determine threshold deltas
        list_threshold_deltas.append(
            new_rates_thresholds - np.append(0, new_rates_thresholds[:-1])
            )

        # determine current and new rates when threshold is reached
        sorted_diff_sign_idxs = diff_sign_idxs[argsort_new_rate_threshold]
        rates = [sum_abs_z - 2 * sum(abs_z[sorted_diff_sign_idxs, 0])]
        for _idx in sorted_diff_sign_idxs:
            rates.append(rates[-1] + 2 * abs_z[_idx, 0])
        list_rates.append(np.array(rates))

    # determine upperbounds (UBs) for gammas in Phase 1
    phase_1_gammas_UBs = np.empty((n, 1))
    for idx, rates in enumerate(list_rates):
        last_idx_neg_rate = np.argwhere(rates >= 0)[0, 0] - 1
        thresholds = list_rates_thresholds[idx]
        if last_idx_neg_rate >= 0:
            x_iter_upper_bound = min(u[idx, 0], thresholds[last_idx_neg_rate])
            max_gamma_decrease = - find_gamma_for_x_iter_increase(
                idx, list_rates, list_rates_thresholds,
                list_threshold_deltas, x_iter,
                x_iter_upper_bound - x_iter[idx]
                )
            phase_1_gammas_UBs[idx, 0] = max_gamma_decrease
        else:
            phase_1_gammas_UBs[idx, 0] = 0

    # phase 1: decrease v_i's as much as possible in order of v_i

    # determine whether all v_i's can be decreased as much as possible
    v_LB = v_iter - phase_1_gammas_UBs
    gammas = det_gammas_decreases(v_LB, v_iter, np.min(v_LB))
    x_iter_increase_info = [
        find_x_iter_increase_for_gamma(
            i, list_rates, list_rates_thresholds,
            list_threshold_deltas, x_iter, gamma
            )
        for i, gamma in enumerate(gammas)
        ]
    x_iter_increases = np.array([info[0] for info in x_iter_increase_info])
    possible_val_increase = np.dot(mu_goal.T, x_iter_increases)[0][0]
    possible_value = val_mu_goal_T_x_iter + possible_val_increase

    if possible_value < goal_value:
        # process decrease and continue to phase 2

        x_iter += x_iter_increases
        v_iter -= gammas
        val_mu_goal_T_x_iter += possible_val_increase

        for i in range(n):
            list_rates[i] = x_iter_increase_info[i][1]
            list_rates_thresholds[i] = x_iter_increase_info[i][2]
            list_threshold_deltas[i] = x_iter_increase_info[i][3]

    elif possible_value == goal_value:
        # goal reached

        x_iter += x_iter_increases

        return (l + x_iter).dot(z.T)

    else:
        # do not process decrease and start binary search for right decrease

        # init binary search
        LB_cut = np.min(v_LB)
        UB_cut = np.max(v_iter)
        count = 0

        while ((goal_value - val_mu_goal_T_x_iter) / goal_value
               > BINSEARCH_PRECISION and count < MAX_BINSEARCH_ITER):

            count += 1
            cut = (LB_cut + UB_cut) / 2

            # determine possible value for cut
            gammas = det_gammas_decreases(v_LB, v_iter, cut)
            x_iter_increase_info = [
                find_x_iter_increase_for_gamma(
                    i, list_rates, list_rates_thresholds,
                    list_threshold_deltas, x_iter, gamma
                    )
                for i, gamma in enumerate(gammas)
                ]
            x_iter_increases = np.array(
                [info[0] for info in x_iter_increase_info]
                )
            possible_val_increase = np.dot(mu_goal.T, x_iter_increases)[0][0]
            possible_value = val_mu_goal_T_x_iter + possible_val_increase

            if possible_value < goal_value:
                # process decrease

                UB_cut = cut
                x_iter += x_iter_increases
                v_iter -= gammas
                val_mu_goal_T_x_iter += possible_val_increase

                for i in range(n):
                    list_rates[i] = x_iter_increase_info[i][1]
                    list_rates_thresholds[i] = x_iter_increase_info[i][2]
                    list_threshold_deltas[i] = x_iter_increase_info[i][3]

            elif possible_value == goal_value:
                # goal reached

                x_iter += x_iter_increases

                return (l + x_iter).dot(z.T)

            else:
                # do not process decrease and update lower bound

                LB_cut = cut

        if count == MAX_BINSEARCH_ITER:
            print(
                f'Warning: could not reach precision {BINSEARCH_PRECISION}, '
                f'after {MAX_BINSEARCH_ITER} iterations. Current relative '
                f'difference is: '
                f'{(goal_value - val_mu_goal_T_x_iter) / goal_value}'
                )

        return (l + x_iter).dot(z.T)

    # phase 2: all rates are non-negative (for those i: x_iter[i] < u[i]),
    # rate-proportionally increase v_i's in order from small to large

    # try to increase all v_i's to their maximum (to get an upper bound)
    v_max = np.array(
        [v_iter[i] + find_gamma_for_x_iter_increase(
            i, list_rates, list_rates_thresholds,
            list_threshold_deltas, x_iter, u[i] - x_iter[i]
            )
         for i in range(n)]
        )
    gammas = det_gammas_increase(v_iter, v_max, np.max(v_max))
    x_iter_increase_info = [
        find_x_iter_increase_for_gamma(
            i, list_rates, list_rates_thresholds,
            list_threshold_deltas, x_iter, gamma
            )
        for i, gamma in enumerate(gammas)]
    x_iter_increases = np.array([info[0] for info in x_iter_increase_info])
    possible_val_increase = np.dot(mu_goal.T, x_iter_increases)[0][0]
    possible_value = val_mu_goal_T_x_iter + possible_val_increase

    if possible_value < goal_value:

        raise Exception("Impossible...")

    elif possible_value == goal_value:
        # goal reached

        x_iter += x_iter_increases

        return (l + x_iter).dot(z.T)

    else:
        # do not process increase and start binary search

        LB_cut = np.min(v_iter)
        UB_cut = np.max(v_max)
        count = 0

        while ((goal_value - val_mu_goal_T_x_iter) / goal_value
               > BINSEARCH_PRECISION and count < MAX_BINSEARCH_ITER):

            count += 1
            cut = (LB_cut + UB_cut) / 2

            # determine possible value for cut
            gammas = det_gammas_increase(v_iter, v_max, cut)
            x_iter_increase_info = [
                find_x_iter_increase_for_gamma(
                    i, list_rates, list_rates_thresholds,
                    list_threshold_deltas, x_iter, gamma
                    )
                for i, gamma in enumerate(gammas)]
            x_iter_increases = np.array(
                [info[0] for info in x_iter_increase_info]
                )
            possible_val_increase = np.dot(mu_goal.T, x_iter_increases)[0][0]
            possible_value = val_mu_goal_T_x_iter + possible_val_increase

            if possible_value < goal_value:
                # process increase

                LB_cut = cut
                x_iter += x_iter_increases
                v_iter += gammas
                val_mu_goal_T_x_iter += possible_val_increase

                for i in range(n):
                    list_rates[i] = x_iter_increase_info[i][1]
                    list_rates_thresholds[i] = x_iter_increase_info[i][2]
                    list_threshold_deltas[i] = x_iter_increase_info[i][3]

            elif possible_value == goal_value:
                # goal reached

                x_iter += x_iter_increases

                return (l + x_iter).dot(z.T)

            else:
                # do not process increase and update upper bound

                UB_cut = cut

        if count == MAX_BINSEARCH_ITER:
            print(
                f'Warning: could not reach precision {BINSEARCH_PRECISION}, '
                f'after {MAX_BINSEARCH_ITER} iterations. Current relative '
                f'difference is: '
                f'{(goal_value - val_mu_goal_T_x_iter) / goal_value},'
                f' returning current solution found.'
                )

        return (l + x_iter).dot(z.T)


def index_set_for_new_edges(G, z_min, z_plus, l, u):
    """Returns a list with set indexes for which x_i (aka x_iter) should be 0
    to ensure that no new edges appear. It is also written in the article as
    S_0^supp. """

    S = []

    for i in range(len(G)):

        zeros = (G[[i], :] == 0).T
        x_increase_slack = (z_plus > 0) * (u[i, 0] > 0)
        x_decrease_slack = (z_min > 0) * (l[i, 0] < 0)
        option_1 = max(np.logical_and(zeros, x_increase_slack))
        option_2 = max(np.logical_and(zeros, x_decrease_slack))

        if option_1 or option_2:
            S.append(i)

    return S


def min_norm_rank_1_pert_pres_stoch(
    G, mu, mu_goal, normtype='inf',
    u=None, sameSuppPrecision=None
    ):
    """Under that assumption that

        mu_goal.T u >= 1

    it finds an explicit solution to:

        min || Delta ||
        s.t.
            mu_goal^T (G + Delta) = mu_goal^T
            G + Delta >= 0
            rank(Delta) = 1

    This is an elaboration of the algorithms from Appendix D to solve the
    problems from Theorem 6.2.

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Probability vector of the stationary distribution goal.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    u : np.array
        The u vector from the article, see determine_u(). This can be
        given so that it can be reused for speed reasons.
    sameSuppPrecision : None or float
        If this is a float, we will find only solutions that do not change the
        support, i.e., for which supp(G) = supp(G + Delta). The float value
        is the minimal value of (G + Delta)(i, j) in case G(i,j) > 0. If it is
        None, we are allowed to modify the support.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    # init
    if u is None:
        u = determine_u(G, mu_goal)
    z, z_plus, z_min = determine_z(G, mu_goal)

    if sameSuppPrecision is not None:
        # hack situation so that we only find solutions with same support as G

        S_0_supp = index_set_for_new_edges(G, z_min, z_plus, 0 * u, u)
        u[u > 0] -= sameSuppPrecision
        u[S_0_supp, 0] = 0

    if mu_goal.T.dot(u) < 1 - PRECISION:
        raise Exception(
            f'This function only works when mu_goal.T.dot(u) >= 1, '
            f'its value is now {mu_goal.T.dot(u)}'
            )
    elif mu_goal.T.dot(u) == 1:
        DeltaSol = u.dot(z.T) / (mu_goal.T.dot(u))
        return DeltaSol
    elif normtype == 'inf':
        # start construction of an optimal solution for inf-norm

        # init
        x_iter = np.array(u)  # start decreasing from upper bound
        val_mu_goal_T_x_iter = mu_goal.T.dot(x_iter)

        while val_mu_goal_T_x_iter > 1:

            max_val = np.max(x_iter)
            max_idxs = np.argwhere(x_iter == max_val)[:, 0]
            non_max_x_iter_vals = x_iter[x_iter < max_val]
            if len(non_max_x_iter_vals) > 0:
                second_max_val = np.max(non_max_x_iter_vals)
            else:
                second_max_val = 0
            total_mu_goal_max = np.sum(mu_goal[max_idxs])

            # determine delta: the maximum decrease
            delta = max_val - second_max_val
            if val_mu_goal_T_x_iter - total_mu_goal_max * delta < 1:
                # adjust delta to decrease as much as possible
                delta = (val_mu_goal_T_x_iter - 1) / total_mu_goal_max

            # update x_iter
            x_iter[max_idxs] -= delta
            val_mu_goal_T_x_iter -= total_mu_goal_max * delta

    elif normtype == '1':
        # start construction of an optimal solution for 1-norm

        # init
        x_iter = np.zeros((len(mu), 1))  # start increasing from zeros vector
        val_mu_goal_T_x_iter = 0
        mu_goal_sorted = np.argsort(mu_goal, axis=0)[::-1]
        count = 0

        while val_mu_goal_T_x_iter < 1 - PRECISION:

            cur_index = mu_goal_sorted[count]

            # determine delta: the maximum increase for current index
            delta = u[cur_index, 0]
            mu_goal_cur_index = mu_goal[cur_index, 0]
            if val_mu_goal_T_x_iter + mu_goal_cur_index * delta > 1:
                # adjust delta to decrease as much as possible
                delta = (1 - val_mu_goal_T_x_iter) / mu_goal_cur_index

            # update x_iter
            x_iter[cur_index, 0] += delta
            val_mu_goal_T_x_iter += mu_goal_cur_index * delta

            count += 1

    else:
        raise NotImplementedError

    DeltaSol = x_iter.dot(z.T)

    return DeltaSol


def min_norm_rank_1(P, mu_goal, normtype='inf'):
    """Finds a Delta that solves:

        min || Delta ||
        s.t.
            mu_goal^T (P + Delta) = mu_goal^T
            rank(Delta) = 1

    Note that there is no guarantee that (P + Delta) >= 0.
    This is an elaboration of Corollary 5.3 that follows from Theorem 5.1.

    Parameters
    ----------
    P : np.array
        Stochastic matrix.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1', '2', or 'inf'.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    # init
    n = len(P)
    I = np.eye(n)

    # determine the x vector
    if normtype in ['1', 1, 'one']:
        x = np.zeros((n, 1))
        x[np.argmax(mu_goal)] = 1
    elif normtype in ['2', 2, 'two']:
        x = np.array(mu_goal)
    elif normtype in ['inf', np.inf]:
        x = np.ones((n, 1))
    else:
        raise TypeError(f'Normtype {normtype} not implemented.')

    DeltaSol = x.dot(mu_goal.T).dot(I - P) / (mu_goal.T.dot(x))

    return DeltaSol


def goal_MC_optimization(
    G, mu_goal, normtype='inf', rankOne=False,
    ensureNonNeg=True, onlyRowIdxs=None,
    verbose=True, DeltaHist=None, sameSuppPrecision=None,
    timeLimit=60
    ):
    """Find the best perturbation Delta for P so that
            mu_goal^T (G + Delta) = mu_goal^T,
    i.e., so that mu_goal becomes its new stationary distribution.

    The problem is:
        min || Delta ||
        s.t.
            mu_goal^T (G + Delta) = mu_goal^T
            (P + Delta) >= 0    [optional: see ensureNonNeg]
            rank(Delta) = 1    [optional: see rankOne]

    If DeltaHist is given (not None), the optimization will also take the
    history of Delta's into account. In particular, the problem solved when
    DeltaHist is not None is:

            min || DeltaHist + Delta ||
        s.t.
            mu_goal^T (G + Delta) = mu_goal^T
            (P + Delta) >= 0    [optional: see ensureNonNeg]
            rank(Delta) = 1    [optional: see rankOne]

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    rankOne : bool
        If True, a rank 1 Delta will be sought.
    ensureNonNeg : bool
        If True, ensures non-negativity of G + Delta, else not.
    onlyRowIdx : list or None
        If a list is given, it will only adjust the corresponding rows in Delta.
    verbose : bool
        If True, reports are printed, else nothing.
    DeltaHist : np.array
        Sum of previous Deltas that can be taken into account in the opt.
    sameSuppPrecision : None or float
        If this is a float, we will only find solutions that do not change the
        support, i.e., for which supp(G) = supp(G + Delta). The float value
        is the minimal value of (G + Delta)(i, j) in case G(i,j) > 0. If it is
        None, we are allowed to modify the support.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    # technical init
    n = len(G)
    if DeltaHist is None:
        DeltaHist = np.zeros((n, n))

    # init model
    ILP = pulp.LpProblem("ILP_best_perturbation_given_goal", pulp.LpMinimize)

    # decision variables
    Delta = [[pulp.LpVariable(f'Delta({i},{j})') for j in range(n)]
             for i in range(n)]  # Delta[i][j] = perturbation of P[i, j]
    DeltaAbs = [
        [pulp.LpVariable(f'DeltaAbs({i},{j})', lowBound=0) for j in range(n)]
        for i in range(n)
        ]  # kind of hacky: DeltaAbs[i][j] = |Delta[i][j] + DeltaHist[i][j]|
    objVar = pulp.LpVariable(
        f'objective_variable',
        lowBound=0
        )  # varies based on chosen norm

    # objective
    ILP += objVar, "objective"

    # constraints:
    # ============

    # 1) ensure that mu_goal is reached
    for j in range(n):
        ILP += sum(
            [mu_goal[i, 0] * (G[i, j] + Delta[i][j]) for i in range(n)]
            ) == mu_goal[j, 0], f'set_mu_goal{j}'

    if ensureNonNeg:
        # 2) ensure that new Markov chain is non-negative
        for i in range(n):
            for j in range(n):
                ILP += G[i, j] + Delta[i][j] >= 0, f'(P + Delta)({i},{j}) >= 0'

    # 3) ensure that DeltaAbs gets its property
    for i in range(n):
        for j in range(n):
            ILP += DeltaAbs[i][j] >= Delta[i][j] + DeltaHist[i][j], \
                   f'DeltaAbs({i},{j}) >= Delta({i},{j})'
            ILP += DeltaAbs[i][j] >= - Delta[i][j] - DeltaHist[i][j], \
                   f'DeltaAbs({i},{j}) >= - Delta({i},{j})'

    # 4) set objective variable based on norm type
    if normtype == '1':
        # maximum absolute column sum
        for j in range(n):
            ILP += objVar >= sum([DeltaAbs[i][j] for i in range(n)])
    elif normtype == 'inf':
        # maximum absolute row sum
        for i in range(n):
            ILP += objVar >= sum([DeltaAbs[i][j] for j in range(n)])
    else:
        raise NotImplementedError(f'Normtype {normtype} is not implemented.')

    if rankOne:
        # ensure that a rank 1 solution is found
        colVals = [pulp.LpVariable(f'colVals({i})') for i in range(n)]
        rowVals = mu_goal.T.dot(np.eye(n) - G)  # ensures constraint 5) holds
        for i in range(n):
            for j in range(n):
                ILP += Delta[i][j] == colVals[i] * rowVals[0, j], \
                       f'Ensure that Delta is rank 1: ({i},{j})'
    else:
        # 5) ensure that row sum is 1
        for i in range(n):
            ILP += sum([G[i, j] + Delta[i][j] for j in range(n)]) == 1, \
                   f'Row_sum_{i} == 0'

    if onlyRowIdxs is not None:
        for i in range(n):
            if i in onlyRowIdxs:
                continue
            for j in range(n):
                ILP += Delta[i][j] == 0, f'Delta({i},{j}) = 0'

    if sameSuppPrecision is not None:
        for i in range(n):
            for j in range(n):
                if G[i, j] > 0:
                    ILP += G[i, j] + Delta[i][j] >= sameSuppPrecision, \
                           f'Ensure (P + Delta)({i},{j}) >= sameSuppPrecision'
                elif G[i, j] == 0:
                    ILP += G[i, j] + Delta[i][j] == 0, \
                           f'Ensure (P + Delta)({i},{j}) == 0'
                else:
                    raise Exception(f'P({i},{j}) < 0!')

    # solve problem
    ILP.solve(pulp.GUROBI(msg=verbose, timeLimit=timeLimit))

    if ILP.status != 1:
        print(
            'Gurobi did not find an optimum,' +
            f' its status number = {ILP.status}' +
            f' which means: {pulp.LpStatus[ILP.status]}.'
            f' An infeasible solution is returned.'
            )

        return np.zeros((n, n))  # return an infeasible solution

    if verbose:
        print("\nMILP report:")
        print("Optimization status:", pulp.LpStatus[ILP.status])
        print("Objective value =", pulp.value(ILP.objective))

    # retrieve and return solution
    DeltaSol = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            DeltaSol[i, j] = Delta[i][j].varValue

    return DeltaSol


def existence_rank_1_Delta_pres_stochasticity(G, mu_goal, u=None):
    """Checks whether mu_delta^T u >= 1, which means that there exists a rank-1
    perturbation Delta that preserves stochasticity.

    There is an option to give u to speed up computation.
    """

    if u is None:
        return mu_goal[:, 0].dot(determine_u(G, mu_goal)[:, 0]) >= 1
    else:
        return mu_goal[:, 0].dot(u[:, 0]) >= 1


def max_convex_softened_mu_goal(G, mu, mu_goal):
    """Denote the convex softened mu_goal as

        mu(alpha) := (1 - alpha) mu + alpha mu_goal.

    It will find the maximum alpha and corresponding mu(alpha) for which the
    problem from min_norm_rank_1_pert_pres_stoch() can be solved.
    """

    max_alpha = min(
        1, max_feas_stepsize_pres_stochasticity(G, mu, mu_goal)
        )  # if > 1 we can even shoot past our mu_goal, which we do not want

    return max_alpha, (1 - max_alpha) * mu + max_alpha * mu_goal


def max_feas_stepsize_pres_stochasticity(G, mu, mu_goal):
    """Returns the maximum scaling factor alpha^* which is discussed after
    Theorem 6.1, it denotes the maximal step size alpha towards (alpha * mu_goal
    + (1-alpha) * mu) such that there is a rank-1 perturbation Delta for which
    (G + Delta) >= 0. """

    u = determine_u(G, mu_goal)
    d = determine_d(mu, mu_goal)

    return (mu.flatten().dot(u) / (1 - d.flatten().dot(u)))[0]


def softened_mu_goal(mu, mu_goal, indexes_set):
    """Softens the mu_goal to soft_mu_goal for which it holds that
        soft_mu_goal[indexes_set] = mu_goal[indexes_set]
    and the rest of soft_mu_goal is relatively equal to mu. Input
    indexes_set can be an int or an iterable. """

    if hasattr(indexes_set, '__len__'):
        if len(indexes_set) == len(mu):
            return mu_goal  # assuming that indexes in set are all unique

    soft_mu_goal = np.array(mu)
    remaining_mass_to_divided = (1 - np.sum(mu_goal[indexes_set]))
    soft_mu_goal *= remaining_mass_to_divided / (1 - np.sum(mu[indexes_set]))
    soft_mu_goal[indexes_set] = mu_goal[indexes_set]

    return soft_mu_goal


def FR1SH(
    G, mu, mu_goal, normType='inf', verbose=True,
    phi=10 ** (-3), sequence='random',
    doMaxConvexJumpToPiGoal=False,
    jumpToGoalIfPossible=True,
    rankDeltaBeforeDirectJumpToGoal=0,
    considerDeltaHistory=False,
    trackBestSolutionAlongTheWay=True,
    timeLimit=None
    ):
    """Iteratively optimizes

    min || Delta ||
    s.t.
        mu_goal^T (G + Delta) = mu_goal^T
        (G + Delta) >= 0

    by taking rank-1 steps towards softened versions of mu_goal. This is the
    implementation of FR1SH from the paper.

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    verbose : bool
        If True, reports are printed, else nothing.
    phi : float
        Stops when difference between goal and current is smaller than phi.
    sequence : string
        Type of sequence used for the element-wise update.
    doMaxConvexJumpToPiGoal : bool
        If True, we will move as much as possible to the softened mu_goal
        as described in max_convex_softened_mu_goal(). Note that this ensures
        that in case it can be solved directly, it will.
    jumpToGoalIfPossible : bool
        If True, the optimization will jump directly to the goal when this is
        possible with a rank 1 perturbation.
    rankDeltaBeforeDirectJumpToGoal : int
        If jumpDirectlyToGoalIfPossible = True, we only jump once the current
        Delta has a rank of rankDeltaBeforeDirectJumpToGoal. If
        jumpDirectlyToGoalIfPossible = False, it is not used.
    considerDeltaHistory : bool
        If True, also the history of Deltas is taken into account when
        finding a minimal norm rank 1 solution.
    trackBestSolutionAlongTheWay : bool
        If True, keep track of the best solutions along the way by checking at
        each step whether we can jump to the goal.
    timeLimit : int
        The method is stopped after timeLimit seconds.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    if verbose:
        print('Start element_wise_rank_1_iter_opt():')

    if timeLimit is None:
        timeLimit = np.inf

    if jumpToGoalIfPossible and \
        existence_rank_1_Delta_pres_stochasticity(G, mu_goal):
        if rankDeltaBeforeDirectJumpToGoal == 0:
            # we can jump right to mu_goal: let's do this
            return min_norm_rank_1_pert_pres_stoch(
                G, mu, mu_goal, normtype=normType
                )

    # technical init
    n = len(mu)
    diff = np.inf
    step = 0
    list_diffs = []
    current_mu = np.array(mu)
    DeltaSol = np.zeros((n, n))
    improvement_found = True
    startTimeExperiment = time.time()
    timeLeft = True

    if trackBestSolutionAlongTheWay:
        # init best solution, possibly with direct jump solution
        u = determine_u(G, mu_goal)  # save for speed reasons
        if existence_rank_1_Delta_pres_stochasticity(G, mu_goal, u=u):
            bestDeltaSol = min_norm_rank_1_pert_pres_stoch(
                G,
                current_mu,
                mu_goal,
                normtype=normType,
                u=u
                )
            normBestDeltaSol = calculate_norm(bestDeltaSol, normType)

            if verbose:
                print(
                    'Init best solution with direct jump solution which has '
                    f'a norm of {normBestDeltaSol}'
                    )

        else:
            bestDeltaSol = np.zeros((n, n))  # best solution found so far
            normBestDeltaSol = np.inf  # norm of best solution found so far

            if verbose:
                print('Init best solution with zeros solution')

    if doMaxConvexJumpToPiGoal:
        step += 1

        alpha, soft_mu_goal = max_convex_softened_mu_goal(
            G, current_mu, mu_goal
            )  # soft_mu_goal can be reached: no need to check it
        DeltaSol += min_norm_rank_1_pert_pres_stoch(
            G, current_mu, soft_mu_goal, normtype=normType
            )
        current_mu = soft_mu_goal  # update current mu

        # save/report results
        diff = np.linalg.norm(soft_mu_goal - mu_goal)
        list_diffs.append(diff)
        if verbose:
            print(f'{step} (max convex jump): diff = {diff}')

    while diff > phi and improvement_found and timeLeft:

        improvement_found = False
        step += 1

        # determine the sequence of element-wise perturbation
        if sequence == 'random':
            idxs_sequence = range(n)
        elif sequence == 'ascending':
            idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())[
                            step >= 2:]  # skip the first, because this is already correct starting at the previous iteration (when step >= 2)
        elif sequence == 'descending':
            idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())[
                            ::-1]
        else:
            raise ValueError('Unknown sequence given.')

        for _idx, i in enumerate(idxs_sequence):

            if time.time() - startTimeExperiment > timeLimit:
                print(
                    f'element_wise_rank_1_iter_opt() reached its '
                    f'time limit of {timeLimit}s.'
                    )
                timeLeft = False
                break

            # determine candidate for softened mu goal
            soft_mu_goal = softened_mu_goal(current_mu, mu_goal, i)

            potential_diff = np.linalg.norm(soft_mu_goal - mu_goal)
            if potential_diff >= diff:
                continue  # no improvement found here

            u = determine_u(
                G + DeltaSol, soft_mu_goal
                )  # save for speed reasons
            if existence_rank_1_Delta_pres_stochasticity(
                G + DeltaSol, soft_mu_goal, u=u
                ):

                # determine DeltaSol
                if considerDeltaHistory:
                    DeltaSol += min_norm_rank_1_pert_pres_stoch_consider_hist(
                        G + DeltaSol, current_mu, soft_mu_goal,
                        normtype=normType, u=u, DeltaHist=DeltaSol
                        )
                else:
                    DeltaSol += min_norm_rank_1_pert_pres_stoch(
                        G + DeltaSol, current_mu, soft_mu_goal,
                        normtype=normType, u=u
                        )

                # update tracking variables
                current_mu = soft_mu_goal  # update current mu
                improvement_found = True
                diff = potential_diff  # this will be the new diff

                # save/report results
                list_diffs.append(diff)
                if verbose:
                    print(
                        f'While loop {step}: diff = {diff} ({round(100 * _idx / n, 4)}%)'
                        )

                if jumpToGoalIfPossible or trackBestSolutionAlongTheWay:

                    u = determine_u(G + DeltaSol, mu_goal)  # save for speed
                    if existence_rank_1_Delta_pres_stochasticity(
                        G + DeltaSol, mu_goal, u=u
                        ):
                        # we can jump right to mu_goal

                        # calculate solution finalDeltaSol
                        if considerDeltaHistory:
                            finalDeltaSol = DeltaSol \
                                + min_norm_rank_1_pert_pres_stoch_consider_hist(
                                    G + DeltaSol, current_mu, mu_goal,
                                    normtype=normType, u=u, DeltaHist=DeltaSol
                                    )
                        else:
                            finalDeltaSol = DeltaSol \
                                + min_norm_rank_1_pert_pres_stoch(
                                    G + DeltaSol, current_mu, mu_goal,
                                    normtype=normType, u=u
                                    )

                        if jumpToGoalIfPossible and (
                            np.linalg.matrix_rank(
                                DeltaSol
                                ) >= rankDeltaBeforeDirectJumpToGoal):
                            # we can jump right to mu_goal: let's do this
                            return finalDeltaSol

                        if trackBestSolutionAlongTheWay:

                            normFinalDeltaSol = calculate_norm(
                                finalDeltaSol,
                                normType
                                )

                            if normFinalDeltaSol < normBestDeltaSol:
                                # found new best
                                normBestDeltaSol = normFinalDeltaSol
                                bestDeltaSol = finalDeltaSol

                                if verbose:
                                    print(
                                        f'Found new best norm of value '
                                        f'{normBestDeltaSol} and rank '
                                        f'{np.linalg.matrix_rank(bestDeltaSol)}'
                                        )
            else:
                continue

    # try to complete the solution
    u = determine_u(G + DeltaSol, mu_goal)  # save for speed reasons
    if existence_rank_1_Delta_pres_stochasticity(G + DeltaSol, mu_goal, u=u):

        if verbose:
            print('We can jump from the stepwise solution to goal.')

        if considerDeltaHistory:
            DeltaSol += min_norm_rank_1_pert_pres_stoch_consider_hist(
                G + DeltaSol, current_mu, mu_goal, normtype=normType, u=u,
                DeltaHist=DeltaSol
                )
        else:
            DeltaSol += min_norm_rank_1_pert_pres_stoch(
                G + DeltaSol, current_mu, mu_goal, normtype=normType, u=u
                )
    else:
        # iterations did not lead to a feasible solution

        if verbose:
            print('FR1SH did not lead to a feasible solution.')

        if trackBestSolutionAlongTheWay:
            if check_feasibility_solution(G, bestDeltaSol, mu_goal):
                # the best solution found along the way is feasible
                if verbose:
                    print(
                        'Along the way a feasible solution was found. '
                        'This will be returned'
                        )

                return bestDeltaSol
            else:
                print(
                    'Warning: FR1SH could not find a feasible DeltaSol. '
                    'An infeasible solution returned'
                    )
        else:
            print(
                'Warning: FR1SH could not find a feasible DeltaSol. '
                'An infeasible solution returned'
                )

        return DeltaSol

    if trackBestSolutionAlongTheWay:
        # it is certain that feasible solutions are found, return best found

        if verbose:

            normDeltaSol = calculate_norm(DeltaSol, normType)

            if normBestDeltaSol < normDeltaSol:
                print(
                    "The last DeltaSol was not the best one "
                    f"(best norm = {normBestDeltaSol} and last norm found "
                    f"is = {normDeltaSol})!"
                    )
            else:
                print(
                    "The best DeltaSol was the last one: "
                    f"best norm = {normBestDeltaSol} and last norm found "
                    f"is = {normDeltaSol}"
                    )

        return bestDeltaSol

    else:

        return compare_Delta_with_direct_jump(
            DeltaSol,
            G,
            mu,
            mu_goal,
            normType=normType,
            verbose=verbose
            )


def R1SH(
    G, mu, mu_goal, normType='inf', verbose=True,
    sequence='random',
    jumpDirectlyToGoalIfPossible=True,
    rankDeltaBeforeDirectJumpToGoal=0,
    considerDeltaHistory=False,
    trackBestSolutionAlongTheWay=True,
    timeLimit=None
    ):
    """Iteratively optimizes

    min || Delta ||
    s.t.
        mu_goal^T (G + Delta) = mu_goal^T
        (G + Delta) >= 0

    by taking rank-1 steps towards softened version of mu_goal. THis is the
    implementation of R1SH from the paper.

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    verbose : bool
        If True, reports are printed, else nothing.
    sequence : string
        Type of sequence used for the element-wise update.
    jumpDirectlyToGoalIfPossible : bool
        If True, the optimization will jump directly to the goal when this is
        possible with a rank 1 perturbation.
    rankDeltaBeforeDirectJumpToGoal : int
        If jumpDirectlyToGoalIfPossible = True, we only jump once the current
        Delta has a rank of rankDeltaBeforeDirectJumpToGoal. If
        jumpDirectlyToGoalIfPossible = False, it is not used.
    considerDeltaHistory : bool
        If True, also the history of Deltas is taken into account when
        finding a minimal norm rank 1 solution.
    trackBestSolutionAlongTheWay : bool
        If True, keep track of the best solutions along the way by checking at
        each step whether we can jump to the goal.
    timeLimit : int
        The method is stopped after timeLimit seconds.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    if verbose:
        print('Start new_element_wise_rank_1_iter_opt():')

    if timeLimit is None:
        timeLimit = np.inf

    # technical init
    n = len(mu)
    list_diffs = [np.linalg.norm(mu - mu_goal)]
    current_mu = np.array(mu)
    DeltaSol = np.zeros((n, n))
    taboo_idxs = []
    startTimeExperiment = time.time()

    if trackBestSolutionAlongTheWay:
        # init best solution, possibly with direct jump solution
        u = determine_u(G, mu_goal)  # save for speed reasons
        if existence_rank_1_Delta_pres_stochasticity(G, mu_goal, u=u):
            bestDeltaSol = min_norm_rank_1_pert_pres_stoch(
                G,
                current_mu,
                mu_goal,
                normtype=normType,
                u=u
                )
            normBestDeltaSol = calculate_norm(bestDeltaSol, normType)

            if verbose:
                print(
                    'Init best solution with direct jump solution which has '
                    f'a norm of {normBestDeltaSol}'
                    )

        else:
            bestDeltaSol = np.zeros((n, n))  # best solution found so far
            normBestDeltaSol = np.inf  # norm of best solution found so far

            if verbose:
                print('Init best solution with zeros solution')

    # determine the sequence of element-wise perturbation
    if sequence == 'random':
        idxs_sequence = range(n)
    elif sequence == 'ascending':
        idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())
    elif sequence == 'descending':
        idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())[::-1]
    elif sequence == 'mu_goal^T_u':
        u = determine_u(G, mu_goal)
        idxs_sequence = np.argsort((mu_goal * u).flatten())
    else:
        raise ValueError('sequence not recognized.')

    for index, i in enumerate(idxs_sequence[:-1]):

        if time.time() - startTimeExperiment > timeLimit:
            print(
                f'new_element_wise_rank_1_iter_opt() reached its '
                f'time limit of {timeLimit}s.'
                )
            break

        # determine candidate for softened mu goal
        fix_idxs = [x for x in idxs_sequence[:index + 1] if x not in taboo_idxs]
        soft_mu_goal = softened_mu_goal(current_mu, mu_goal, fix_idxs)

        # try to jump to soft_mu_goal
        u = determine_u(G + DeltaSol, soft_mu_goal)  # save for speed reasons
        if existence_rank_1_Delta_pres_stochasticity(
            G + DeltaSol, soft_mu_goal, u=u
            ):

            if verbose:
                print(f'Fixating index {i} ({100 * index / n}%)')

            # determine DeltaSol
            if considerDeltaHistory:

                fastDeltaSol = min_norm_rank_1_pert_pres_stoch_consider_hist(
                    G + DeltaSol, current_mu, soft_mu_goal,
                    normtype=normType,
                    u=u, DeltaHist=DeltaSol
                    )

                DeltaSol += fastDeltaSol
            else:
                DeltaSol += min_norm_rank_1_pert_pres_stoch(
                    G + DeltaSol, current_mu, soft_mu_goal,
                    normtype=normType,
                    u=u
                    )

            # update tracking variables
            current_mu = soft_mu_goal  # update current mu
            list_diffs.append(np.linalg.norm(current_mu - mu_goal))

            if jumpDirectlyToGoalIfPossible or trackBestSolutionAlongTheWay:

                u = determine_u(G + DeltaSol, mu_goal)  # save for speed reasons
                if existence_rank_1_Delta_pres_stochasticity(
                    G + DeltaSol, mu_goal, u=u
                    ):
                    # we can jump right to mu_goal

                    # calculate solution finalDeltaSol
                    if considerDeltaHistory:
                        finalDeltaSol = DeltaSol \
                            + min_norm_rank_1_pert_pres_stoch_consider_hist(
                                G + DeltaSol, current_mu, mu_goal,
                                normtype=normType, u=u, DeltaHist=DeltaSol
                                )
                    else:
                        finalDeltaSol = DeltaSol \
                            + min_norm_rank_1_pert_pres_stoch(
                                G + DeltaSol, current_mu, mu_goal,
                                normtype=normType, u=u
                                )

                    if jumpDirectlyToGoalIfPossible and (np.linalg.matrix_rank(
                        DeltaSol) >= rankDeltaBeforeDirectJumpToGoal):
                        # we can jump right to mu_goal: let's do this
                        return finalDeltaSol

                    if trackBestSolutionAlongTheWay:

                        normFinalDeltaSol = calculate_norm(
                            finalDeltaSol,
                            normType
                            )

                        if normFinalDeltaSol < normBestDeltaSol:
                            # found new best
                            normBestDeltaSol = normFinalDeltaSol
                            bestDeltaSol = finalDeltaSol

                            if verbose:
                                print(
                                    f'Found new best norm of value '
                                    f'{normBestDeltaSol} and rank '
                                    f'{np.linalg.matrix_rank(bestDeltaSol)}'
                                    )

        else:
            # adding i to the set was problematic, reconsider at the end
            if verbose:
                print(
                    f'Warning: Index {i} could not be fixed. Added to taboo'
                    f' list and continue the element wise fixation.'
                    )
            taboo_idxs.append(i)

    numb_elems_fixed = len(fix_idxs)
    if numb_elems_fixed < n - 1:

        print(
            'Warning: R1SH could not fix all elements and therefore did not '
            f'lead to a feasible solution. Only {numb_elems_fixed} elements out'
            f' of {n - 1} could be fixed.'
            )

        if trackBestSolutionAlongTheWay:
            if check_feasibility_solution(G, bestDeltaSol, mu_goal):

                print(
                    'Along the way a feasible solution was found, '
                    'this will now be returned.'
                    )

                return bestDeltaSol

            else:

                print(
                    'R1SH did not find a feasible solution was found. '
                    'An infeasible solution will be returned.'
                    )

                return DeltaSol

    # at least one feasible solution found
    if trackBestSolutionAlongTheWay:
        # return best solution found

        if verbose:

            normDeltaSol = calculate_norm(DeltaSol, normType)

            if normBestDeltaSol < normDeltaSol:
                print(
                    "The last DeltaSol was not the best one "
                    f"(best norm = {normBestDeltaSol} and last norm found "
                    f"is = {normDeltaSol})!"
                    )
            else:
                print(
                    "The best DeltaSol was the last one: "
                    f"best norm = {normBestDeltaSol} and last norm found "
                    f"is = {normDeltaSol}"
                    )

        return bestDeltaSol

    else:

        return compare_Delta_with_direct_jump(
            DeltaSol,
            G,
            mu,
            mu_goal,
            normType=normType,
            verbose=verbose
            )


def compare_Delta_with_direct_jump(
    Delta, G, mu, mu_goal, normType='inf', verbose=False
    ):
    """Compares the given Delta with the solution found by
    min_norm_rank_1_pert_pres_stoch() if it exists, this last solution
    is denoted by directDelta. The best solution will be returned.

    Parameters
    ----------
    Delta : np.array
        The Delta to compare the directDelta with.
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    verbose : bool
        If True, reports are printed, else nothing.

    Returns
    -------
    directDelta or Delta : np.array
        The solution with smallest norm.
    """

    u = determine_u(G, mu_goal)  # save for speed reasons
    if existence_rank_1_Delta_pres_stochasticity(G, mu_goal, u=u):
        # direct jump solution exists

        if verbose:
            print('Direct jump solution exists. Compare with Delta.')

        # determine direct jump solution
        directDelta = min_norm_rank_1_pert_pres_stoch(
            G,
            mu,
            mu_goal,
            normtype=normType,
            u=u
            )
        normDirectDelta = calculate_norm(directDelta, normType)

        # return best solution
        normDelta = calculate_norm(Delta, normType)
        if normDirectDelta <= normDelta:

            if verbose:
                print('Direct jump solution is best and returned.')

            return directDelta
        else:

            if verbose:
                print('Delta is best and returned.')

            return Delta

    else:

        if verbose:
            print('Direct jump solution does not exist, return Delta.')

        return Delta


def RISH_K(
    G, mu, mu_goal, normType='inf', verbose=True,
    sequence='random',
    jumpDirectlyToGoalIfPossible=True,
    rankDeltaBeforeDirectJumpToGoal=0,
    considerDeltaHistory=False,
    numbIntervals=2
    ):
    """Iteratively optimizes

    min || Delta ||
    s.t.
        mu_goal^T (G + Delta) = mu_goal^T
        (G + Delta) >= 0

    by taking rank-1 steps towards softened versions of mu_goal. The softened
    versions of mu_goal are determined by dividing the goal into numbIntervals
    intervals. This is the implementation of RISH(K) from the paper.

    Parameters
    ----------
    G : np.array
        Stochastic matrix.
    mu : np.array
        Stationary distribution of G.
    mu_goal : np.array
        Goal stationary distribution.
    normtype : str
        String indication the norm type to apply. Options: '1' or 'inf'.
    verbose : bool
        If True, reports are printed, else nothing.
    sequence : string
        Type of sequence used for the element-wise update.
    jumpDirectlyToGoalIfPossible : bool
        If True, the optimization will jump directly to the goal when this is
        possible with a rank 1 perturbation.
    rankDeltaBeforeDirectJumpToGoal : int
        If jumpDirectlyToGoalIfPossible = True, we only jump once the current
        Delta has a rank of rankDeltaBeforeDirectJumpToGoal. If
        jumpDirectlyToGoalIfPossible = False, it is not used.
    considerDeltaHistory : bool
        If True, also the history of Deltas is taken into account when
        finding a minimal norm rank 1 solution.
    numbIntervals : int
        Number of intervals used.

    Returns
    -------
    DeltaSol : np.array
        The perturbation matrix.

    """

    # technical init
    n = len(mu)
    list_diffs = [np.linalg.norm(mu - mu_goal)]
    current_mu = np.array(mu)
    DeltaSol = np.zeros((n, n))

    # determine the sequence of element-wise perturbation
    if sequence == 'random':
        idxs_sequence = list(range(n))
    elif sequence == 'ascending':
        idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())
    elif sequence == 'descending':
        idxs_sequence = np.argsort(np.abs(current_mu - mu_goal).flatten())[::-1]
    elif sequence == 'mu_goal^T_u':
        u = determine_u(G, mu_goal)
        idxs_sequence = np.argsort((mu_goal * u).flatten())
    else:
        raise Exception('Unknown sequence type.')

    # determine intervals
    interval_widths = math.floor(n / numbIntervals)
    interval_boundaries = [i * interval_widths
                           for i in range(1, numbIntervals + 1)]
    not_all_idxs_covered = (n % numbIntervals) > 0
    if not_all_idxs_covered:
        interval_boundaries[-1] = n  # last interval includes a bit more

    for boundary in interval_boundaries:

        # determine candidate for softened mu goal
        fix_idxs = [x for x in idxs_sequence[:boundary]]
        soft_mu_goal = softened_mu_goal(current_mu, mu_goal, fix_idxs)

        u = determine_u(G + DeltaSol, soft_mu_goal)  # save for speed reasons
        if existence_rank_1_Delta_pres_stochasticity(
            G + DeltaSol, soft_mu_goal, u=u
            ):

            # determine DeltaSol
            if considerDeltaHistory:

                fastDeltaSol = min_norm_rank_1_pert_pres_stoch_consider_hist(
                    G + DeltaSol, current_mu, soft_mu_goal,
                    normtype=normType,
                    u=u, DeltaHist=DeltaSol
                    )
                DeltaSol += fastDeltaSol
            else:
                DeltaSol += min_norm_rank_1_pert_pres_stoch(
                    G + DeltaSol, current_mu, soft_mu_goal,
                    normtype=normType,
                    u=u
                    )

            # update tracking variables
            current_mu = soft_mu_goal  # update current mu
            list_diffs.append(np.linalg.norm(current_mu - mu_goal))

            if jumpDirectlyToGoalIfPossible:
                u = determine_u(G + DeltaSol, mu_goal)  # save for speed reasons
                if existence_rank_1_Delta_pres_stochasticity(
                    G + DeltaSol, mu_goal, u=u
                    ):
                    if np.linalg.matrix_rank(DeltaSol) >= \
                        rankDeltaBeforeDirectJumpToGoal:
                        # we can jump right to mu_goal: let's do this
                        if considerDeltaHistory:
                            DeltaSol += \
                                min_norm_rank_1_pert_pres_stoch_consider_hist(
                                    G + DeltaSol, current_mu, mu_goal,
                                    normtype=normType, u=u, DeltaHist=DeltaSol
                                    )
                        else:
                            DeltaSol += min_norm_rank_1_pert_pres_stoch(
                                G + DeltaSol, current_mu, mu_goal,
                                normtype=normType, u=u
                                )
                        return DeltaSol
        else:
            # adding i to the set was problematic, reconsider at the end
            print(f'interval_rank_1_iter_opt with {numbIntervals} did not work')
            return DeltaSol

    if not check_feasibility_solution(G, DeltaSol, mu_goal):

        # try jumping to goal
        u = determine_u(G + DeltaSol, mu_goal)  # save for speed reasons
        if existence_rank_1_Delta_pres_stochasticity(
            G + DeltaSol, mu_goal, u=u
            ):
            if considerDeltaHistory:
                DeltaSol += min_norm_rank_1_pert_pres_stoch_consider_hist(
                    G + DeltaSol, current_mu, mu_goal, normtype=normType,
                    u=u, DeltaHist=DeltaSol
                    )
            else:
                DeltaSol += min_norm_rank_1_pert_pres_stoch(
                    G + DeltaSol, current_mu, mu_goal, normtype=normType, u=u
                    )
            return DeltaSol

        idxs_still_not_reached = [x for x in range(n) if x not in fix_idxs]
        print(
            f'For {sequence}, could reach the indexes: {idxs_still_not_reached}'
            )

    return compare_Delta_with_direct_jump(
        DeltaSol,
        G,
        mu,
        mu_goal,
        normType=normType,
        verbose=verbose
        )
