
"""
This is the mainfile to run the clustering algorithm, and include some helpful
plots of the data.
"""
# TODO modify to run as a script from the terminal, and add 3D plotting. Probably
# write separate functions for plotting 2D and 3D stuff. Also need to add plots of
# eigenthings
# TODO move most of the code to the individual clustering files
# TODO sanitize input to the plotting functions

from spectral_clustering import *
from lib.gen_data import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

"""
Available functions to generate and cluster data (* indicates optional
arguments.)

generate_blobs(n, num_centers*)
    Uses sklearn.datasets.make_blobs to generate random normally distributed
    data around a number of centers given by num_centers. Data is more spread
    out than in default arguments to generate_gaussian_data

generate_circles(n, noise*, factor*)
    Generates data with noise around two concentric circles. The argument factor
    (which must be between 0 and 1) gives the ratio between the radii of the
    inner and outer circle

generate_moons(n, noise*)
    Generates data around two half moons, with given noise

generate_curve_data(n, curves*, bounds*, noise*)
    Generates data around curves given by functions defined in the array curves.
    Bounds represents the bounds for the x and y coordinates, and noise
    represents the standard deviation for the gaussian noise added
"""

fig = plt.figure()
fig2 = plt.figure()
data_plot = fig.add_subplot(131, aspect='equal')
clustered = fig.add_subplot(132, aspect='equal')
spectrum = fig.add_subplot(133, adjustable='box')
plt.subplots_adjust(bottom = 0.2)

n = 1000

#  while True:
    #  shape = input("""Pick a shape to generate synthetic data in:
    #  1: Gaussians
    #  2: 2 Concentric Circles
    #  3: 2 half-moons
    #  4: Two predefined curves
#  >>> """)
    #  try:
        #  shape = int(shape)
        #  if shape < 5 and 0 < shape:
            #  break
    #  except:
        #  print("Invalid input. Please try again")
        #  pass

shape = 1

data = {
        1 : generate_blobs,
        2 : generate_circles,
        3 : generate_moons,
        4 : generate_poly_data,
        }[shape](n)

x = []
y = []

# separate out the points
for point in data:
    x.append(point[0])
    y.append(point[1])
 
# plot the data
data_plot.scatter(x, y, color='gray', marker='o', linewidth='0', s = 15)

# do some clustering. The number of clusters is bounded by 10 right now.
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'purple', \
        'pink', 'orange', 'black']

print()
while True:
    num_clusters = input("Enter the number of clusters to make (max 10)\n> ")
    try:
        num_clusters = int(num_clusters)
        break
    except:
        print("\n---\nPlease enter a number\n---\n\n")

num_clusters = min(10, num_clusters)

clusters, cluster_inds, evals, U_trans = spec_cluster_data(data, num_clusters)

# plot the clusters
for i in range(num_clusters):

    x = []
    y = []

    for point in clusters[i]:
        x.append(point[0])
        y.append(point[1])

    clustered.scatter(x, y, color=colors[i], marker='o', linewidth='0', s = 15)

#  adjust bounds of the spectrum plot
max_eval = evals[6][0]
min_eval = evals[0][0]
spectrum.set_ylim(bottom=min_eval*0.5, top=max_eval*1.1)

#  plot the spectrum
points = list(zip(range(6), evals))
x = [point[0] for point in points[:6]]
y = [point[1][0] for point in points[:6]]
spectrum.scatter(x, y, color='blue', marker='o', linewidth='1', s = 30)

#  fig2 = plt.figure()

#  Make the eigenvector plots
tups = list(combinations(range(num_clusters), 2))
plots = []
for i in range(len(tups)):
    subplot_index = 100 + 10 * len(tups) + i + 1
    s_1 = U_trans[tups[i][0]]
    str_1 = "$s_" + str(tups[i][0]) + "$"
    s_2 = U_trans[tups[i][1]]
    str_2 = "$s_" + str(tups[i][1]) + "$"
    points = list(zip(s_1, s_2))

    plots.append(fig2.add_subplot(subplot_index, \
            xlabel = str_1, ylabel = str_2))
    # create the clusters
    clusts = [[] for i in range(num_clusters)]
    for i in range(n):
        (clusts[cluster_inds[i]]).append(points[i])

    # now plot the clusters
    for i in range(num_clusters):
        x = []
        y = []
        for point in clusts[i]:
            x.append(point[0])
            y.append(point[1])

        (plots[-1]).scatter(x, y, color=colors[i], s = 15, marker='o', linewidth='0')

fig.tight_layout()
fig2.tight_layout()
plt.show()


