"""
File containing functions to generate Swiss Roll data set. The data set is
generated as described here:

    https://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html
"""

import math
import numpy

def generate_swiss_roll():
    """
    Generates the swiss roll data set by sampling in 2D around for Gaussians
    centered at (7.5, 7.5), (7.5, 12.5), (12.5, 12.5) and (12.5, 7.5), with the
    identity as the covariance matrix. 400 points are generated around each
    Gaussian for a total of 1600 points. 

    Returns a data array and an array containing the colormap values, so the
    points can be nicely plotted
    """

    centers = [(7.5, 7.5), (7.5, 12.5), (12.5, 7.5), (12.5, 12.5)]
    data = dict()
    for i in range(4):
        data[i] = numpy.random.multivariate_normal(centers[i], \
                [[1,0],[0,1]], 400)

    # append everything to get a nice list
    for i in range(1, 4):
        data[0] = numpy.append(data[0], data[i], axis=0)
    pre_data = data[0]

    # get the colormap
    cs = numpy.array([point[0]**2 for point in pre_data])

    # get the data points
    data = numpy.array([(x*math.cos(x), y, x*math.sin(x)) for \
        (x, y) in pre_data])

    return [data, cs]
