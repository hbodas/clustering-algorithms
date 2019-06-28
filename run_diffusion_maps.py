
"""
File to run the diffusion maps algorithm. Generates data, plots it if possible,
and then runs the algorithm. Currently only works for the Swiss Roll dataset
(see lib for a function to generate this data set).
"""
from lib.make_swiss_roll import *
from diffusion_maps import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

# first generate the data and plot it
data, cs = generate_swiss_roll(5000)
xs = [point[0] for point in data]
ys = [point[1] for point in data]
zs = [point[2] for point in data]

ax.scatter(xs, ys, zs=zs, zdir='z', c=cs, cmap="gist_rainbow")

#  # create a new figure
ax2 = fig.add_subplot(122)

em_data = embed_data(data, dim=2, t=1)
x = [point[0] for point in em_data]
y = [point[1] for point in em_data]

ax2.scatter(x, y, c=cs, cmap="gist_rainbow")

plt.show()
