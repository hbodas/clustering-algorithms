"""
File that contains all data generation functions.

Every function draws a total of n samples, and returns a numpy array of the
sampled points, where each point is represented by a numpy array. 
"""

import numpy, math
from sklearn.datasets import make_blobs, make_moons, make_circles
from lib.make_poly_curves import *

def generate_blobs(n, num_centers=3):
    """
    Uses the make_blobs function in the sklearn.datasets library to generate
    normally distributed ''blobs'' with randomly selected centers and random
    distribution. 
    """
    # sample the data
    data = make_blobs(n_samples = n, n_features = 2, centers = num_centers, \
            cluster_std = 1.8, center_box = (-10, 10), shuffle = True, \
            random_state = None)[0]

    print("Generated {} data points around {} means".format(n, num_centers,\
        flush=True))
    return data

def generate_circles(n, noise=0.07, factor=0.40):
    """
    Generates data sample around two concentric circles if radius given by the
    ration factor, and with noise given by noise.
    """
    # sample the data
    data = make_circles(n_samples = n, shuffle = True, noise = noise, \
            random_state = None, factor = factor)[0]

    print("Generated {} data points in concentric circles".format(n, flush=True))
    return data

def generate_moons(n, noise = 0.07):
    """
    Generates data sample around two half moons, with noise given by noise.
    """
    # sample the data
    data = make_moons(n_samples = n, shuffle = True, noise = noise, \
            random_state = None)[0]

    print("Generated {} data points around double moons".format(n, flush=True))
    return data

def generate_poly_data(n, curves=None, bounds=(-10, 10), noise=1):
    """
    Generates data along curves defined in the array curves
    """
    if curves == None:
        curves = [lambda x : 8+math.cos(x), lambda x : 2*math.sin(x) + 0.5*x - 4]

    data = gen_2D_poly_data(n, curves, bounds, noise)

    return data


