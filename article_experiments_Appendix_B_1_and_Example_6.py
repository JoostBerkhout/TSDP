"""
Created on 11/22/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: The experiments from Appendix B.1 and Example 6 are done here.
"""

import utilities
from MC_generators import MMsK_queue
from TSDP_optimization import *

# global init
from utilities import (
    calc_stationary_distribution, solution_report,
    determine_z, determine_u, determine_l,
    )

normType = 'inf'
numb_decimals = 4

# instance init
s = 2
K = 1
arr_rate = 1
ser_rate = 1.8

# technical init
numb_states = s + K + 1
G = MMsK_queue(s, K, arr_rate, ser_rate)
mu_Delta = np.ones((numb_states, 1)) / numb_states

# calculate measures
mu = calc_stationary_distribution(G)
Delta = min_norm_rank_1(G, mu_Delta, normtype=normType)

# report results
print('\nFor Appendix B.1:\n')
print(f'mu^T =\n{mu.T.round(numb_decimals)}')
print(f'Delta =\n{Delta.round(numb_decimals)}')
print(f'G + Delta =\n{(G + Delta).round(numb_decimals)}')
solution_report(G, Delta, mu_Delta)
print('In Latex:')
print('Delta = ')
utilities.bmatrix(Delta)
print('G + Delta = ')
utilities.bmatrix(G + Delta)

# Example 6
# =========

print('\nExample 6:\n')

# instance init
s = 2
K = 1
arr_rate = 1
ser_rate = 1.8

# technical init
numb_states = s + K + 1
G = MMsK_queue(s, K, arr_rate, ser_rate)
mu_G = calc_stationary_distribution(G)

# choose a mu_Delta relatively close to mu_G
mu_Delta = np.ones((numb_states, 1)) / numb_states

# calculate measures
z, z_plus, z_min = determine_z(G, mu_Delta)
l = determine_l(G, mu_Delta)
u = determine_u(G, mu_Delta)

# allowed to change support
Delta = min_norm_rank_1_pert_pres_stoch(
    G,
    mu_G,
    mu_Delta,
    normtype=normType
    )

# report results
print('\nFor Example 6:\n')
print(f'mu^T =\n{mu.T.round(numb_decimals)}')
print(f'Delta =\n{Delta.round(numb_decimals)}')
print(f'G + Delta =\n{(G + Delta).round(numb_decimals)}')
solution_report(G, Delta, mu_Delta)
print('In Latex:')
print('Delta = ')
utilities.bmatrix(Delta)
print('G + Delta = ')
utilities.bmatrix(G + Delta)
