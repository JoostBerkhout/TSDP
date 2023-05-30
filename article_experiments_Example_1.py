"""
Created on 11/22/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: Experiments for Example 1 from the article are done here.
"""

from TSDP_optimization import *

# global init
from utilities import calc_stationary_distribution

normType = 'inf'
numb_decimals = 3

# instance init
G = np.array(
    [[1 / 3, 2 / 3],
     [3 / 4, 1 / 4]]
    )
mu_Delta = np.array(
    [[1 / 2],
     [1 / 2]]
    )

# calculate measures
mu = calc_stationary_distribution(G)
Delta = min_norm_rank_1(G, mu_Delta, normtype=normType)

# report results
print('\nFor Example 1:\n')
print(f'mu^T =\n{mu.T.round(numb_decimals)}')
print(f'Delta =\n{Delta.round(numb_decimals)}')
print(f'G + Delta =\n{(G + Delta).round(numb_decimals)}')
