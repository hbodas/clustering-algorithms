
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

def embed_data(data, **kwargs):
    """
    Finds an embedding of the data using the diffusion maps algorithm. k is the
    kernel function.

    Keyword arguments: 
    - dim: The number of dimensions to embed in. Must be specified if delta
      isn't. Takes priority over delta when both are specified.
    - delta: The accuracy parameter. Must be specified if dim isn't
    - alpha: The parameter alpha for the anisotropic kernel. Defaults to 1

    Returns an array representing the embedded data.
    """
    # check arguments
    try:
        kwargs['dim'] or kwargs['delta']
    except:
        raise KeyError("Please enter either the number of dimensions or the " + \
        "accuracy parameter delta as a keyword argument")

    # look at other keyword arguments
    try:
        alpha = kwargs['alpha']
    except:
        alpha = 0

    try:
        t = kwargs['alpha']
    except:
        t = 2

    num_points = len(data)
    NUM_POINTS = range(num_points) # an iterator

    # construct the similarity graph and normalize as needed
    K = construct_graph(data)
    V = [sum(K[:,i]) for i in NUM_POINTS]
    D_inv_al = numpy.diag([V[i]**(-alpha) for i in NUM_POINTS])
    K_al = numpy.matmul(D_inv_al, numpy.matmul(K, D_inv_al))
    V_al_inv = numpy.diag([1/sum(K_al[:,i]) for i in NUM_POINTS])
    M = numpy.matmul(V_al_inv, K_al)

    # compute eigenvalues and right eigenvectors of M, and sort them
    evals, evects = numpy.linalg.eig(M)

    # normalize the eigenvectors
    evects = list(map(lambda x : x / norm(x), [evects[:, i] for i in \
        NUM_POINTS]))

    spectrum = sorted(zip(evals, [evects[i] for i in NUM_POINTS]), key=lambda\
            x:x[0], reverse=True)

    # compute the number of dimensions to embed in, if not already provided.
    try: em_dim = kwargs['dim']
    except:
        delta = kwargs['delta'] # we have already checked that this exists
        em_dim = 1
        while spectrum[em_dim][0]**t > delta*spectrum[1][0]**t:
            em_dim += 1

    # find the data points
    points = []
    for i in NUM_POINTS:
        points.append([])
        for j in range(em_dim):
            k = j+1
            points[i].append(spectrum[k][0]*spectrum[k][1][i])

    return [points, spectrum]
