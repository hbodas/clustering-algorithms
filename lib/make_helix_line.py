"""
Contains function to generate data along a helix and a straight line. Generates
half the points along the helix and half the points on the line.
"""

import numpy

def make_helix(n=2000):

    # set parameters
    r = 0.1 # the radius of the helix
    height = 15 # height
    sigma = 0.0 # noise strength

    # create the helix
    #  phi = height*numpy.random.rand(3*n//4)
    #  n_1 = sigma*numpy.random.rand(3*n//4)
    #  n_2 = sigma*numpy.random.rand(3*n//4)
    phi = height*numpy.random.rand(3*n//4)
    n_1 = sigma*numpy.random.rand(3*n//4)
    n_2 = sigma*numpy.random.rand(3*n//4)
    X = (r+n_1) * numpy.cos(phi)
    Y = (r+n_1) * numpy.sin(phi)
    Z = 1.5*phi
    helix = numpy.array([X, Y, Z]).transpose()
    helix_colors = list(phi**0.7)

    # create the line
    phi = height*numpy.random.rand(n//4)
    n_1 = sigma*numpy.random.rand(n//4)
    n_2 = sigma*numpy.random.rand(n//4)
    n_3 = sigma*numpy.random.rand(n//4)
    X = n_1
    Y = r+n_2
    Z = 15 + 2*phi + n_3

    line = numpy.array([X, Y, Z]).transpose()
    line_colors = list(phi**0.7)
    #  line = []
    #  line_colors = []

    return [helix, helix_colors, line, line_colors]

