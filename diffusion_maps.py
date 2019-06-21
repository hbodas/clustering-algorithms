
"""
Implements a non-linear dimension reduction algorithm to find an embedding of
data points in a high dimension to a lower dimensional manifold. Based on the
discussion in the paper:

Coifman, Ronald R., and Stéphane Lafon. "Diffusion maps." Applied and
computational harmonic analysis 21.1 (2006): 5-30.

This implementation does not implement the full algorithm, the number of
dimensions to embed in passed be passed in as an argument.
"""

import numpy, math
from lib.similarity_graph import *

def matrix_power(M, t):
    """ 
    Computes the matrix power M^t recursively. Note: t must be a non-negative
    integer. Returns M^t as a numpy array.
    """
    if t == 0: return M
    else: return numpy.matmul(M, matrix_power(M, t-1))

def embed_data(data, em_dim=2, t=1, alpha=1):
    """
    Finds an embedding of the data into the number of dimensions given by
    em_dim. Runs the main diffusion maps algorithm with parameter alpha for the
    anisotropic kernel, and t time steps.

    Returns an array representing the embedded data.
    """
    # first construct the similarity graph
    W = construct_graph(data)
    D_neg_alpha = numpy.diag([1 / (sum(W[:, i]))**alpha for i in range(num_points)])

    # find the normalized Markov matrix
    W_alpha = numpy.matmul(D, numpy.matmul(W, D))
    D_alpha_inv = numpy.diag([1 / sum(W_alpha[:, i]) for i in range(num_points)])
    M = numpy.matmul(D_alpha_inv, W_alpha)

    # find the appropriate power of the Markov matrix and compute its spectrum
    M_t = matrix_power(M, t)
    x, V = numpy.linalg.eig(M_t)
    x = sorted(zip(x, range(len(x))), key=lambda x : x[0], reverse=True)

    em_mat = [] 
    for i in range(em_dim):
        eig = V[x[i][1]]
        em_mat.append((x[i][0]**t)*eig)

    em_mat = numpy.array(em_mat)

    return list(map(lambda x : numpy.matmul(em_mat, x), data))

def make_plots():
    pass
