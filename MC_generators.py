"""
Created on 18/05/2023

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Contains some generators for Markov chains (MCs).
"""
import numpy as np


def ring(N, b):
    """Returns transition matrix of a ring network (function by Bernd). """

    A_ring = np.zeros((N, N))

    for i in range(N):
        A_ring[i, i] = 1 - 2 * b

    for i in range(1, N - 1):
        A_ring[i, i - 1] = b
        A_ring[i, i + 1] = b

    A_ring[0, 1] = b
    A_ring[0, N - 1] = b

    A_ring[N - 1, 0] = b
    A_ring[N - 1, N - 2] = b

    return A_ring


def star(N, beta, gamma):
    """Returns transition matrix of a star network (function by Bernd). """

    A_star = np.zeros((N, N))
    A_star[0, 0] = 1 - beta

    for i in range(1, N):
        A_star[0, i] = beta / (N - 1)

    for i in range(1, N):
        A_star[i, i] = gamma

    for i in range(1, N):
        A_star[i, 0] = 1 - gamma

    return A_star


def MMsK_queue(s, K, arr_rate, ser_rate):
    """Returns the transition matrix of a M/M/s/K queue where s denotes
    the number of service places and K the number of buffer places. The
    arrival rate is denoted by arr_rate and the service rate at every service
    place by ser_rate. """

    # init
    numb_states = s + K + 1
    Q = np.zeros((numb_states, numb_states))

    # construct first the (continuous-time Markov chain) Q matrix
    # ===========================================================

    # set first and last row of Q
    Q[0, 1] = arr_rate
    Q[-1, -2] = s * ser_rate

    # set middle rows
    for state in range(1, numb_states - 1):
        Q[state, state - 1] = min(state, s) * ser_rate
        Q[state, state + 1] = arr_rate

    # ensure that rows sum to 0
    row_sums = np.sum(Q, axis=1)
    np.fill_diagonal(Q, - row_sums)

    # transform Q to subordinated chain with P
    # ========================================

    scaling = np.max(row_sums)  # Nicole Leder: scaling > max(row_sums)
    P = np.eye(numb_states) + Q / scaling

    return P
