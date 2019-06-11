"""
File that contains functions for generation of noisy data along univariate
polynomial curves. Evauluates polynomials at x coordinates selected uniformly at
random from the given range, and adds a gaussian noise to the curves. Generates
a total of n points.
"""

# TODO add ways to change the noise distribution as well as the x coordinate
# distribution

import numpy

def gen_2D_poly_data(n, curves, bounds, noise=0.10):
    """
    Generates noisy data along the polynomial curves. Does so by first
    evaluating each polynomial at a random point, and then adds a random amount
    of gaussian noise to the y-coordinate. 

    Generates a total of n samples. The array curves contains functions around
    which the noisy data is generated. Bounds is a tuple that contains the
    maximum and minimum x coordinate. noise is the standard deviation of the
    gaussian that is used to generate noise.
    """
    lo, hi = bounds

    if len(curves) == 0:
        print("Provide at least one curve to generate around!")
        return

    # initialize the points array
    points = []

    for i in range(n):

        # pick a random x coordinate (uniformly)
        x = numpy.random.uniform(low = lo, high = hi, size = None)
        offset = numpy.random.normal(loc=0.0, scale=noise, size=None)
        index = i % len(curves)
        points.append(numpy.array([x, curves[index](x) + offset]))

    return numpy.array(points)
