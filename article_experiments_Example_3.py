"""
Created on 11/22/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Article experiments for Example 3 are done here.
"""

import utilities
from MC_generators import ring, star

# global init
from TSDP_optimization import min_norm_rank_1
from utilities import (
    calc_stationary_distribution, calculate_norm, determine_z,
    determine_u, determine_l,
    )

normType = 'inf'
numb_decimals = 3

# Example 3
# =========

print('\nFor Example 3\n')

# Experiment 1:

# init ring network
n = 4
b = .3
G = ring(n, b)

# init star network
beta = .9
gamma = .9
G_Delta = star(n, beta, gamma)

# technical init
mu = calc_stationary_distribution(G)
mu_Delta = calc_stationary_distribution(G_Delta)
z, z_plus, z_min = determine_z(G, mu_Delta)
l = determine_l(G, mu_Delta)
u = determine_u(G, mu_Delta)

# consider "a" Delta as specified in Theorem 2
a_Delta = u.dot(z.T) / (mu_Delta.T.dot(u)[0][0])
norm_a_Delta = calculate_norm(a_Delta, normType)

# calculate minimum norm
Delta_no_stoch = min_norm_rank_1(
    G,
    mu_Delta,
    normtype=normType
    )
norm_Delta_no_stoch = calculate_norm(Delta_no_stoch, 1)

# report results
print('Experiment 1:')
print('Delta from Theorem 6.1 is:')
utilities.bmatrix(a_Delta)
print(f'\|Delta\|_1 = {calculate_norm(a_Delta, 1)}')
print(f'\min\|Delta\|_1 = {norm_Delta_no_stoch}')

# Experiment 2:

# init star network 1
n = 4
beta_1 = .2
gamma_1 = .9
G = star(n, beta_1, gamma_1)

# init star network 2
beta_2 = .3
gamma_2 = .3
G_Delta = star(n, beta_2, gamma_2)

# technical init
mu = calc_stationary_distribution(G)
mu_Delta = calc_stationary_distribution(G_Delta)
z, z_plus, z_min = determine_z(G, mu_Delta)
l = determine_l(G, mu_Delta)
u = determine_u(G, mu_Delta)

# consider "a" Delta as specified in Theorem 2
a_Delta = u.dot(z.T) / (mu_Delta.T.dot(u)[0][0])
norm_a_Delta = calculate_norm(a_Delta, normType)

# calculate minimum norm
Delta_no_stoch = min_norm_rank_1(
    G,
    mu_Delta,
    normtype='1'
    )
norm_Delta_no_stoch = calculate_norm(Delta_no_stoch, 1)

# report results
print('Experiment 2:')
print('Delta from Theorem 6.1 is:')
utilities.bmatrix(a_Delta)
print(f'\|Delta\|_1 = {calculate_norm(a_Delta, 1)}')
print(f'\min\|Delta\|_1 = {norm_Delta_no_stoch}')
