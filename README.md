# Article code of "Perturbation and Inverse Problems of Stochastic Matrices"

In this repository, the article code can be found of the paper "Perturbation and Inverse Problems of Stochastic Matrices"
written by Joost Berkhout, Paul van Dooren and Bernd Heidergott that appeared in 2023 in SIMAX.
In this paper, the main focus is on solving the target stationary distribution problem (TSDP).

In the TSDP, we are given an irreducible stochastic matrix G with stationary distribution µ > 0
and some target stationary distribution µ'. The goal is to find a perturbation ∆ of the minimum
norm such that G+∆ remains stochastic and has the desired target stationary distribution µ'.

The repository contains all Python scripts that were used to generate the numerical experiment results.
Each script that starts with ``article_`` is dedicated to one or two experiments from the article and is named accordingly.
The rest of the Python code are modules that are used by the article scripts.

## How to use this code?

Download the Python code and run the scripts in your own Python >= 3.6 environment (for example, using [Anaconda](https://www.anaconda.com/download)).
Ensure that [Gurobi](https://www.gurobi.com/) is installed.

The results from the more extensive experiments from Section 8 may differ a bit because of different reasons:

1) Different machine speeds can lead to different results;
2) The specific Python version can affect the results
   (for example, different Barabási–Albert preferential attachment networks are generated with Networkx for the same seed in Python 3.6 and Python 3.9).

## License

This code is under [MIT licence](https://choosealicense.com/licenses/mit/).

## Contact

Joost Berkhout via [joost.berkhout@vu.nl](mailto://joost.berkhout@vu.nl).