"""Quantize the Lab Color Space"""
import matplotlib.pyplot as plt
import numpy as np
import math
import random
%matplotlib inline

reweighting_vector = np.load('../preprocessing/reweighting_vector.npy')
pts_in_hull = np.load('../network/pts_in_hull.npy')

SIZE = 130/10
viz = np.zeros((SIZE*2, SIZE*2))
for i in range(len(pts_in_hull)):
    a, b = pts_in_hull[i]
    frequency = reweighting_vector[i]
    viz[SIZE + a/10][SIZE + b/10] = frequency

print min(reweighting_vector), max(reweighting_vector)

plt.imshow(viz, cmap='hot', interpolation='nearest', extent=[-SIZE, SIZE, -SIZE, SIZE])
plt.colorbar()
plt.suptitle('Frequency Weightings over AB Space', fontsize=15)
plt.ylabel('a', fontsize=10)
plt.xlabel('b', fontsize=10)
plt.show()
