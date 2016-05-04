# -*- coding: utf-8 -*-
"""
Created on Tue May 03 20:30:01 2016

@author: Spindola
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets

np.random.seed(10)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data 
Y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf() 
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
mean_x = np.mean(X,axis=0)
nX = X - mean_x
nXT = nX.T

cov_mtx = np.dot(nXT,nX)/(X.shape[0]-1)
eigen_val_mtx, eigen_vect_mtx = np.linalg.eig(cov_mtx)

cov_mtx2 = eigen_vect_mtx[:,[0,1,2]]

new_data = np.dot(cov_mtx2.T,nXT)

new_dataT = new_data.T

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(new_dataT[Y == label, 0].mean(),
              new_dataT[Y == label, 1].mean() + 1.5,
              new_dataT[Y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

Y = np.choose(Y, [1, 2, 0]).astype(np.float)
ax.scatter(new_dataT[:, 0], new_dataT[:, 1], new_dataT[:, 2], c=Y, cmap=plt.cm.spectral)

#for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
   # ax.text3D(X[Y == label, 0].mean(),
              #X[Y == label, 1].mean() + 1.5,
             # X[Y == label, 2].mean(), name,
              #horizontalalignment='center',
              #bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

#Y = np.choose(Y, [1, 2, 0]).astype(np.float)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=plt.cm.spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.figure(1)

plt.show()