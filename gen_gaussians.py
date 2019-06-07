
"""
File that contains code to randomly sample data from Gaussians
"""

import numpy, math

def generate_gaussian_data(n, means=[[0,0], [10,6], [6,-5]], cov=[[3,0],[0,3]]):
    """
    Generates a sample of data drawn from a number of Gaussian distributions
    with means given in the array means, and covariance matrix given by cov.
    Draws a total n samples.

    Returns a numpy array of the sampled points, where each point is represented
    by a numpy array.
    """
    # validate the arguments passed in
    dimensions = len(means[0])

    # sample the data
    data = None
    for i in range(n):
        j = numpy.random.randint(0,len(means))
        new_data = [numpy.random.multivariate_normal(means[j], cov)]
        if i == 0:
            data = numpy.array(new_data)
        else:
            data = numpy.append(data, new_data, axis = 0)

    print("Generated {} data points around {} means".format(n, len(means),\
        flush=True))
    return data
