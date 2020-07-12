
"""
File to run the diffusion maps algorithm. Generates data, plots it if possible,
and then runs the algorithm. 
"""
from lib.make_swiss_roll import *
from lib.make_helix_line import *
from diffusion_maps import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

n = 800 # the number of points

# first generate the data and plot it (for swiss roll)
roll, cs = generate_swiss_roll(n)
xs = [point[0] for point in roll]
ys = [point[1] for point in roll]
zs = [point[2] for point in roll]
ax.scatter(xs, ys, zs=zs, zdir='z', c=cs, cmap="gist_rainbow", s=15)

# generate data and plot it
#  helix, helix_cs, line, line_cs = make_helix(n)
#  helix_xs = [point[0] for point in helix]
#  helix_ys = [point[1] for point in helix]
#  helix_zs = [point[2] for point in helix]
#  line_xs = [point[0] for point in line]
#  line_ys = [point[1] for point in line]
#  line_zs = [point[2] for point in line]
#  ax.scatter(helix_xs, helix_ys, zs=helix_zs, zdir='z', \
        #  c=helix_cs, s=15, cmap="autumn")
#  ax.scatter(line_xs, line_ys, zs=line_zs, zdir='z', c=line_cs, \
        #  s=15, cmap="winter")


# for swiss roll
em_data, spectrum = embed_data(roll, dim=2, t=1)
x = [point[0] for point in em_data]
y = [point[1] for point in em_data]
ax2.scatter(x, y, c=cs, cmap="gist_rainbow", s=10)

#  # for helix
#  data = list(helix) + list(line)
#  em_data, spectrum = embed_data(data, dim=2, t=1)
#  em_helix_x = [point[0] for point in em_data[:3*n//4]]
#  em_helix_y = [point[1] for point in em_data[:3*n//4]]
#  #  em_helix_y = [0 for point in em_data[:3*n//4]]
#  em_line_x = [point[0] for point in em_data[3*n//4:]]
#  em_line_y = [point[1] for point in em_data[3*n//4:]]
#  #  em_line_y = [0 for point in em_data[3*n//4:]]
#  ax2.scatter(em_helix_x, em_helix_y, c=helix_cs, cmap="autumn", s=15)
#  ax2.scatter(em_line_x, em_line_y, c=line_cs, cmap="winter", s=15)

# plot the spectrum
print([point[0] for point in spectrum[:6]])

plt.show()
