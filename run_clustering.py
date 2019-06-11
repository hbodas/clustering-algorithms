
"""
This is the mainfile to run the clustering algorithm, and include some helpful
plots of the data.
"""
# TODO modify to run as a script from the terminal, and add 3D plotting. Probably
# write separate functions for plotting 2D and 3D stuff. Also need to add plots of
# eigenthings

from spectral_clustering import *
from gen_data import *
import matplotlib.pyplot as plt

"""
Available functions to generate and cluster data (* indicates optional
arguments.)

generate_gaussians(n, means*, cov*)
    Function to generate data around means given in the means array. cov
    represents the covariance matrix used to generate the data in 2D

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
"""


n = 1000
#  data = generate_gaussians(n)
#  data = generate_blobs(n)
#  data = generate_circles(n)
data = generate_moons(n)
#  print(data)

x = []
y = []

# separate out the points
for point in data:
    x.append(point[0])
    y.append(point[1])
 
plt.gca().set_aspect('equal', adjustable='box')

# plot the data
plt.plot(x, y, color='gray', marker='o', linewidth='0')
plt.show()

# clear the figure
plt.clf()
plt.gca().set_aspect('equal', adjustable='box')

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

clusters = spec_cluster_data(data, num_clusters)

for i in range(num_clusters):

    x = []
    y = []

    for point in clusters[i]:
        x.append(point[0])
        y.append(point[1])

    plt.plot(x, y, color=colors[i], marker='o', linewidth='0')

plt.show()
