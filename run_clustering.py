
"""
This is the mainfile to run the clustering algorithm, and include some helpful
plots of the data.
"""

from spectral_clustering import *
from gen_gaussians import *
import matplotlib.pyplot as plt
import sklearn

"""
Available functions to generate and cluster data
"""


n = 1500
data = generate_gaussian_data(n)
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
