
""" 
File that contains functions to perform spectral clustering as described in the
second algorithm in section 4 of this paper:

Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and
computing 17.4 (2007): 395-416.

The code uses k-means clustering provided in by the scikit-learn library

TODO modifiy to use sparse arrays.
"""

import numpy
import math
from sklearn.cluster import KMeans

def norm(x):
    """
    Returns the Euclidean norm of the vector x, represented as a numpy array
    """
    return math.sqrt(sum(map(lambda x : x**2, x)))

def similarity_exp(x, y, sigma = 1.0):
    """
    Returns the similarity of the two points x and y, given by 
    exp((norm(x-y)**2)/(2*sigma**2))
    sigma is a parameter that can be adjusted
    """
    return math.exp(-(norm(x - y)**2)/(2*(sigma**2)))

def find_nearest_neighbors(data, x, num_neighbors):
    """ 
    Finds the k nearest neigbors of the point x and returns a list of indices
    for the list of data points.
    """
    # we want at least one neighbor.
    if num_neighbors == 0: num_neighbors = 1

    # do some sorting by distance
    L = map(lambda y : norm(x - y), data)
    L = zip(L, range(len(data)))
    L = sorted(L, key=lambda x : x[0])

    # find k+1 elements since first element will always be the point itself
    return list(map(lambda x : x[1], L[:num_neighbors+1]))

def construct_graph(data):
    """
    Constructs the similarity graph for the given data and returns the weight
    matrix W.

    Similarity between two points is 0 if neither are of the first few neighbors
    of the other. If not, then a gaussian similarity function is used.

    The number of neighbors is found by taking the log of the number of data
    points
    """
    print("Constructing weight matrix...")

    num_points = len(data)
    W = numpy.zeros((num_points, num_points))
    num_neighbors = math.floor(math.sqrt(num_points))

    for i in range(num_points):
        nearest_neighbors = find_nearest_neighbors(data, data[i], \
                num_neighbors)
        for j in nearest_neighbors:
            W[i][j] = similarity_exp(data[i], data[j])
            W[j][i] = W[i][j]

    return W

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
    
    return clusters
