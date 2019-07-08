"""
File containing functions to generate Swiss Roll data set. The data set is
generated as described here:
"""

import math
import numpy

def generate_swiss_roll(n=400):
    """
    Generates the swiss roll data set by sampling random points in 2D, and then
    using a parametrization to generate points in 3D
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

    swiss_roll = numpy.array([X, Z, Y]).transpose()
    colors = list(map(lambda x:x**2, phi))

    return [swiss_roll, colors]

