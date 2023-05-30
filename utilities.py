"""
Created on 10/22/2020

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Utilities.
"""

import networkx as nx
import numpy as np
from scipy import linalg as la

from constants import PRECISION


def bmatrix(A, precision=4):
    """Returns a LaTeX bmatrix string from a numpy array A. """

    if len(A.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')

    A = np.array(A.round(precision), dtype=object)

    # replace zeros bij int
    A[A == 0] = int(0)

    lines = np.array2string(A, max_line_width=np.infty).replace(
        '[', ''
        ).replace(
        ']', ''
        ).splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    print('\n'.join(rv))


def store_results(
    results_dict, method_name, P, DeltaSol, mu_goal, normType,
    run_time, calc_norm=False
    ):
    """Helper function to store results in results_dict (in place). """

    solution_feasible = check_feasibility_solution(P, DeltaSol, mu_goal)

    if method_name in results_dict:
        results_dict[method_name]['\|Delta\|'].append(
            np.linalg.norm(
                DeltaSol,
                np.inf if normType == "inf" else 1
                ) if solution_feasible or calc_norm else np.nan
            )
        results_dict[method_name]['Feasible'].append(solution_feasible)
        results_dict[method_name]['Run time'].append(run_time)
        results_dict[method_name]['rank(Delta)'].append(
            np.linalg.matrix_rank(
                DeltaSol
                ) if solution_feasible or calc_norm else np.nan
            )
    else:
        results_dict[method_name] = {}
        results_dict[method_name]['\|Delta\|'] = [np.linalg.norm(
            DeltaSol,
            np.inf if normType == "inf" else 1
            )] if solution_feasible or calc_norm else []
        results_dict[method_name]['Feasible'] = [solution_feasible]
        results_dict[method_name]['Run time'] = [run_time]
        results_dict[method_name]['rank(Delta)'] = [np.linalg.matrix_rank(
            DeltaSol
            )] if solution_feasible or calc_norm else []


def find_cliques_from_adjacency_matrix(A):
    """Returns a list of lists where each list gives the clique indexes. """

    posValues = A > 0
    if not (posValues == posValues.T).all():
        print(
            "Warning: Cliques are ill defined for directed graphs."
            " Directed edges are loaded as undirected edges."
            )

    G = nx.from_numpy_matrix(A)

    return [clq for clq in nx.clique.find_cliques(G)]


def calc_stationary_distribution(G):
    n = len(G)
    Z = G.copy() - np.eye(n)
    Z[:, 0] = 1
    e1 = np.zeros((n, 1))
    e1[0] = 1
    return la.solve(Z.transpose(), e1)


def check_feasibility_solution(G, Delta, mu_goal, precision=None):

    if precision is None:
        precision = PRECISION

    stationary_distr = np.max(
        np.abs(mu_goal.transpose().dot(G + Delta) - mu_goal.transpose())
        ) < precision
    non_neg = np.min(G + Delta) >= - precision
    return stationary_distr and non_neg


def calculate_norm(A, normType='inf'):
    """Calculate the norm of A. """

    if normType in ['inf', np.inf]:
        return np.linalg.norm(A, np.inf)
    elif normType in ['one', '1', 1]:
        return np.linalg.norm(A, 1)
    elif normType in ['two', '2', 2]:
        return np.linalg.norm(A, 2)
    else:
        raise NotImplementedError(f'Norm type {normType} not implemented.')


def solution_report(
    G, Delta, mu_goal, numb_decimals=4, rank_precision=10 ** (-5)
    ):
    """Prints a standard report of G, Delta and mu_goal. """

    if np.max(np.isnan(Delta)):
        print('No feasible solution delta found, so no report to print.')
        return
    print(f'Solution report (rounded to {numb_decimals} decimals):')
    print(f'rank(Delta) = {np.linalg.matrix_rank(Delta, rank_precision)}')
    print(
        f'min(G+Delta) = {np.round(np.min(G + Delta), numb_decimals)}'
        )
    print(
        f'max(G+Delta) = {np.round(np.max(G + Delta), numb_decimals)}'
        )
    check_value = np.round(
        np.max(
            np.abs(
                mu_goal.transpose().dot(
                    G + Delta
                    ) - mu_goal.transpose()
                )
            ), numb_decimals
        )
    print(f'The following should be near 0: {check_value}')
    print(f'||Delta||_1 = {np.round(np.linalg.norm(Delta, 1), numb_decimals)}')
    print(f'||Delta||_2 = {np.round(np.linalg.norm(Delta, 2), numb_decimals)}')
    Delta_inf = np.round(np.linalg.norm(Delta, np.inf), numb_decimals)
    print(f'||Delta||_inf = {Delta_inf}')


def determine_d(mu, mu_goal):
    """In terms of the article, it determines vector d. """

    d = mu_goal - mu

    return d


def determine_z(P, mu_goal):
    """In terms of the article, it determines vectors z, z_plus, z_min. """

    n = len(P)
    I = np.eye(n)
    z = mu_goal.T.dot(I - P)  # because mu.T (I - P) = 0, this is removed
    z[np.abs(z) < PRECISION] = 0  # to avoid numerical issues
    z_plus = z * (z > 0)
    z_min = - z * (z < 0)

    return z.T, z_plus.T, z_min.T


def determine_u(P, mu_goal):
    """In terms of the article, it determines vector u. """

    z, z_plus, z_min = determine_z(P, mu_goal)
    z_min_indexes = np.argwhere(z_min > 0)[:, 0]

    if len(z_min_indexes) > 0:
        A = P[:, z_min_indexes] / z_min[z_min_indexes, 0]

        return np.min(A, 1, keepdims=True)
    else:
        return np.zeros((len(P), 1))


def determine_l(P, mu_goal):
    """In terms of the article, it determines vector l. """

    z, z_plus, z_min = determine_z(P, mu_goal)
    z_plus_indexes = np.argwhere(z_plus > 0)[:, 0]

    if len(z_plus_indexes) > 0:
        A = - P[:, z_plus_indexes] / z_plus[z_plus_indexes, 0]

        return np.max(A, 1, keepdims=True)
    else:
        return np.zeros((len(P), 1))


def ergodic_project(mu_goal):
    return np.ones((len(mu_goal), 1)).dot(mu_goal.T)
