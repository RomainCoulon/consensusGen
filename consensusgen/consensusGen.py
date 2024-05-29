# -*- coding: utf-8 -*-
"""
Created on Wed May 29 08:43:09 2024

@author: romain.coulon
"""

import numpy as np
import matplotlib.pyplot as plt

def consensusGen(X, u, *, ng=3, ni=8000):
    """Calculate a reference value using an evolutionary algorithm.
    See: Romain Coulon and Steven Judge 2021 Metrologia 58 065007
    DOI 10.1088/1681-7575/ac31c0

    Parameters
    ----------
    X : list of floats
        Measurement values.
    u : list of floats
        Standard uncertainties of measurement values.
    ng : int, optional
        Number of generations (Default = 3). Set ng=0 for Linear Pool estimation.
    ni : int, optional
        Number of individuals in the whole population (Default = 8000).

    Returns
    -------
    mu : float
        Reference value.
    u_mu : float
        Standard uncertainty of the reference value.
    g0pop : list of floats
        Linear Pool distribution.
    gLpop : list of floats
        EA filtered distribution.
    w : list of floats
        Weights associated with laboratories.
    """

    m = len(X)
    ni_per_group = ni // m  # Number of individuals per group

    if ng == 0:
        generations = 0
    else:
        generations = range(ng)

    group_indices = range(m)
    individual_indices = range(ni_per_group)

    # Initialize matrices and arrays
    Gen2 = np.empty((m, ni_per_group))
    Gen3 = np.empty((m, ni_per_group))
    q = np.empty((m, ni_per_group))
    q2 = np.zeros((m, ni_per_group))
    w = np.ones((m, ni_per_group))

    KCRV = np.empty(ng)  # Mean of the whole population at each generation
    uKCRV = np.empty(ng)  # Standard deviation of the whole population at each generation

    # Generate the first generation
    for i in group_indices:
        q[i] = np.random.normal(X[i], u[i], ni_per_group)

    Q2 = q.ravel()  # Suppress the group separation

    if ng == 0:
        KCRV[0] = np.mean(Q2)
        uKCRV[0] = np.std(Q2) / np.sqrt(m)
        Q = Q2
        Wgth = np.ones(m)
    else:
        # Assign initial genomes to each individual
        for i in group_indices:
            Gen2[i, :] = i + 1
            Gen3[i, :] = i + 1

        for t in generations:
            q2.fill(0)  # Reset q2 for the new generation

            for i in group_indices:
                for j in individual_indices:
                    if w[i, j] != 0:
                        # Find the nearest phenotype
                        Mat1 = np.abs(q[i, j] - q)
                        Mat1[i, j] = float("inf")  # Exclude self
                        nearest_idx = np.unravel_index(np.argmin(Mat1), Mat1.shape)
                        l, c = nearest_idx

                        if Gen2[i, j] != Gen2[l, c]:
                            r = np.random.rand()
                            q2[i, j] = r * q[i, j] + (1 - r) * q[l, c]  # Crossover
                            Gen3[i, j] = np.sqrt(Gen2[i, j] * Gen2[l, c])
                        else:
                            w[i, j] = 0
                            q2[i, j] = q[i, j]
                    else:
                        w[i, j] = 0
                        q2[i, j] = q[i, j]

            q = q2.copy()
            Gen2 = Gen3.copy()
            Q = q.ravel()
            W = w.ravel()
            G = np.where(W != 0)

            Wgth = np.array([np.sum(w[i] != 0) / ni_per_group for i in group_indices])

            KCRV[t] = np.mean(Q[G])
            uKCRV[t] = np.std(Q[G]) / np.sqrt(m)

    mu = KCRV[-1]
    u_mu = uKCRV[-1]
    g0pop = Q2
    gLpop = Q[G]
    w = Wgth / np.sum(Wgth)

    return mu, u_mu, g0pop, gLpop, w

def displayResult(X, u, result, *, lab=False):
    """
    Display the result of the genetic algorithm consensusGen()

    Parameters
    ----------
    X : list of floats
        Measurement values.
    u : list of floats
        Standard uncertainties of measurement values.
    result : list
        Output of consensusGen().
    lab : list, optional
        List of the participants. The default is False.

    Returns
    -------
    None.
    """
    mu, u_mu, g0pop, gLpop, w = result
    nX = len(X)

    print(f"The consensus value is {mu:.4g} ± {u_mu:.2g}")

    if not lab:
        lab = np.linspace(1, nX, nX)
    labstr = [str(int(x)) for x in lab]

    # Data plot
    plt.figure("Data")
    plt.clf()
    plt.errorbar(labstr, X, yerr=u, fmt='ok', capsize=3, ecolor='k', label=r"$x_i$")
    plt.plot(lab - 1, np.ones(nX) * mu, "-r", label=r"$\hat{\mu}$")
    plt.plot(lab - 1, np.ones(nX) * (mu + u_mu), "--r", label=r"$\hat{\mu} + u(\hat{\mu})$")
    plt.plot(lab - 1, np.ones(nX) * (mu - u_mu), "--r", label=r"$\hat{\mu} - u(\hat{\mu})$")
    plt.ylabel(r'Value', fontsize=12)
    plt.xlabel(r'Participant', fontsize=12)
    plt.legend()

    # Weights plot
    plt.figure("Weights")
    plt.clf()
    plt.bar(labstr, w)
    plt.ylabel(r'$w_i$', fontsize=12)
    plt.xlabel(r'Participant', fontsize=12)
    plt.legend()

    # Distributions plot
    plt.figure("Distributions")
    plt.clf()
    plt.hist(g0pop, bins=100, edgecolor='none', density=True, label='Linear Pooling')
    plt.hist(gLpop, bins=100, edgecolor='none', alpha=0.7, density=True, label='Genetic Algorithm')
    plt.ylabel(r'$p(x_i)$', fontsize=12)
    plt.xlabel(r'$x$', fontsize=12)
    plt.legend()

    # Show plots
    plt.show()

# Example usage (replace with actual function call and data):
# X = [your_data]
# u = [your_uncertainties]
# result = consensusGen(X, u)
# displayResult(X, u, result)

