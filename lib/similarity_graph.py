
"""
File that contains functions to generate a similarity matrix for a given data
set. A similarity matrix is constructed using a Gaussian kernel for the k
nearest neighbors (k chosen in code)
"""

import numpy
import math

def norm(x):
    """
    Returns the Euclidean norm of the vector x, represented as a numpy array
    """
    return math.sqrt(sum(map(lambda x : x**2, x)))

def similarity_exp(x, y, sigma = 0.6):
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

def construct_graph(data, sim_fn = similarity_exp, num_neighbors = None):
    """
    Constructs the similarity graph for the given data and returns the weight
    matrix W.

    Similarity between two points is 0 if neither are of the first few neighbors
    of the other. If not, then a gaussian similarity function is used.
    Alternatively, a similarity function can be specified.

    The number of neighbors is found by taking the log of the number of data
    points
    """
    print("Constructing weight matrix...")

    num_points = len(data)
    W = numpy.zeros((num_points, num_points))
    if num_neighbors == None:
        num_neighbors = math.floor(math.log(num_points))
    else:
        num_neighbors = min(num_neighbors, num_points - 1)

    for i in range(num_points):
        nearest_neighbors = find_nearest_neighbors(data, data[i], \
                num_neighbors)
        for j in nearest_neighbors:
            W[i][j] = similarity_exp(data[i], data[j])
            W[j][i] = W[i][j]

    return W

