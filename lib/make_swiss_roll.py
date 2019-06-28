"""
File containing functions to generate Swiss Roll data set. The data set is
generated as described here:

    https://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html
"""

import math
import numpy

def generate_swiss_roll(n=400):
    """
    Generates the swiss roll data set by sampling in 2D around for Gaussians
    centered at (7.5, 7.5), (7.5, 12.5), (12.5, 12.5) and (12.5, 7.5), with the
    identity as the covariance matrix. 400 points are generated around each
    Gaussian for a total of 1600 points. 

    Returns a data array and an array containing the colormap values, so the
    points can be nicely plotted
    """
    # set parameters
    length_phi = 15   #length of swiss roll in angular direction
    length_Z = 15     #length of swiss roll in z direction
    sigma = 0.1       #noise strength

    # create dataset
    phi = length_phi*numpy.random.rand(n)
    xi = numpy.random.rand(n)
    Z = length_Z*numpy.random.rand(n)
    X = 1./6*(phi + sigma*xi)*numpy.sin(phi)
    Y = 1./6*(phi + sigma*xi)*numpy.cos(phi)

    swiss_roll = numpy.array([X, Y, Z]).transpose()
    colors = list(map(lambda x:x**2, phi))

    return [swiss_roll, colors]

