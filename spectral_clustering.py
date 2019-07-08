
""" 
File that contains functions to perform spectral clustering as described in the
second algorithm in section 4 of this paper:

Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and
computing 17.4 (2007): 395-416.

The code uses k-means clustering provided in by the scikit-learn library
"""

# TODO move makeplots function here

import numpy
from lib.similarity_graph import *
from sklearn.cluster import KMeans

def construct_L(W):
    """
    Constructs the random-walk Laplacian matrix as is given in the paper by
    Ulrike. W is the weight matrix. Returns a numpy array representing the
    Laplacian
    """
    print("Constructing the Laplacian")
    num_points = numpy.shape(W)[0]

    # first construct the degree matrix
    D_inv = numpy.diag([1 / sum(W[:, i]) for i in range(num_points)])
    
    return numpy.eye(num_points) - numpy.matmul(D_inv, W)

def spec_cluster_data(data, k):
    """
    Function that does all the clustering. The data points are passed in as the
    argument data, as a numpy array of numpy arrays, and k represents the number
    of clusters.

    Returns a 2D array of clusters.
    """
    num_points = len(data)
    k = min(k, num_points)
    
    W = construct_graph(data)
    L_rw = construct_L(W)

    # find the eigenvectors
    x, V = numpy.linalg.eig(L_rw)

    # give the eigenvalues indices.
    x = sorted(zip(x, range(len(x))), key=lambda x : x[0])

    # compute U
    U_trans = numpy.vstack([V[:, i] for (v, i) in x[:k]])

    kmeans = KMeans(n_clusters = k, n_init = 20).fit(numpy.transpose(U_trans))
    cluster_inds = kmeans.labels_

    # construct the clusters
    clusters = [[] for i in range(k)]

    for i in range(num_points):
        clusters[cluster_inds[i]].append(data[i])
    
    return [clusters, cluster_inds, x, U_trans]
