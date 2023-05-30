"""
Created on 11/28/2021

Author: Joost Berkhout (VU, email: joost.berkhout@vu.nl)

Description: This script prepares the raw data from the folder Raw data.
"""

import numpy as np


def load_adjacency_list(data_path, skip_lines_with_symbol='%'):
    """Returns the adjacency list for the file """

    f = open(data_path, 'r')
    edges_list = []

    for i in f.readlines():

        if skip_lines_with_symbol in i:
            continue

        nodes = i.split(" ")

        if len(nodes) == 2:
            # assume an unweighted network

            n1 = int(nodes[0])
            n2 = int(nodes[1])
            edges_list.append([n1, n2])

        elif len(nodes) == 3:
            # assume a weighted network

            n1 = int(nodes[0])
            n2 = int(nodes[1])
            weight = int(nodes[2])
            edges_list.append([n1, n2, weight])

    f.close()

    return np.array(edges_list)


# prepare university email data from:
# http://konect.cc/networks/arenas-email/

# init
data_path = 'Data\\Raw data\\arenas-email\\out.arenas-email'

# load data
edges_list = load_adjacency_list(data_path, skip_lines_with_symbol='%')

# check data
numb_nodes = edges_list.max()
if numb_nodes != len(np.unique(edges_list)):
    raise Exception('Nodes not labeled from 1, 2, ... , numb_nodes!')

# create adjancency matrix
email_matrix = np.zeros((numb_nodes, numb_nodes))
email_matrix[edges_list[:, 0] - 1, edges_list[:, 1] - 1] = 1
email_matrix += email_matrix.T  # it is an undirected network

# create probability matrix
email_P = np.array(email_matrix)
row_sums_email_matrix = np.sum(email_matrix, axis=1)
for idx, row_sum in enumerate(row_sums_email_matrix):
    email_P[idx, :] /= row_sum

# prepare Moreno health data from:
# http://konect.cc/networks/moreno_health/

# init
data_path = 'Data\\Raw data\\moreno_health\\out.moreno_health_health'

# load data
edges_list = load_adjacency_list(data_path, skip_lines_with_symbol='%')

# check data
numb_nodes = edges_list[:, :2].max()
if numb_nodes != len(np.unique(edges_list)):
    raise Exception('Nodes not labeled from 1, 2, ... , numb_nodes!')

# create adjancency matrix
moreno_matrix = np.zeros((numb_nodes, numb_nodes))
moreno_matrix[edges_list[:, 0] - 1, edges_list[:, 1] - 1] = edges_list[:, 2]

# the probability matrix of Moreno health network is made with own Markov chain
# package

# prepare Euroroads data from:
# http://konect.cc/networks/subelj_euroroad/

# init
data_path = 'Data\\Raw data\\subelj_euroroad\\out.subelj_euroroad_euroroad'

# load data
edges_list = load_adjacency_list(data_path, skip_lines_with_symbol='%')

# check data
numb_nodes = edges_list.max()
if numb_nodes != len(np.unique(edges_list)):
    raise Exception('Nodes not labeled from 1, 2, ... , numb_nodes!')

# create adjancency matrix
road_matrix = np.zeros((numb_nodes, numb_nodes))
road_matrix[edges_list[:, 0] - 1, edges_list[:, 1] - 1] = 1
road_matrix += road_matrix.T  # it is an undirected network

# create probability matrix
road_P = np.array(road_matrix)
row_sums_road_matrix = np.sum(road_matrix, axis=1)
for idx, row_sum in enumerate(row_sums_road_matrix):
    road_P[idx, :] /= row_sum
