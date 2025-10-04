import math
import os
from math import cos
from math import sin
from math import cosh
from math import sinh
from math import acos
from math import asin
from math import acosh
from math import asinh
from math import tan
from math import atan
from math import tanh
from math import atanh
from math import sqrt
from math import pi
from math import floor

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])                # z coordinate of points in inside surface
    return x, y, z


if __name__ == '__main__':
    xAxisLength = 10
    yAxisLength = 10
    zAxisLength = 10

    center = [0, 0, 0]


    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.grid(False)
    ax1._axis3don = False

    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_xlim(-xAxisLength, xAxisLength)
    ax1.set_ylabel(r'$y$', fontsize=12)
    ax1.set_ylim(-yAxisLength, yAxisLength)
    ax1.set_zlabel(r'$z$', fontsize=12)
    ax1.set_zlim(-zAxisLength, zAxisLength)

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    a = Arrow3D([0, xAxisLength], [0, 0], [0, 0], **arrow_prop_dict, color='black')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, yAxisLength], [0, 0], **arrow_prop_dict, color='black')
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, zAxisLength], **arrow_prop_dict, color='black')
    ax1.add_artist(a)

    # Give them a name:
    #ax1.text(0.0, 0.0, -0.2, r'$0$')
    ax1.text(xAxisLength + 0.5, 0.3, 0.5, r'$x$', fontsize=12)
    ax1.text(0.3, yAxisLength + 0.5, 0.5, r'$y$', fontsize=12)
    ax1.text(0.3, 0.3, zAxisLength + 0.5, r'$z$', fontsize=12)

    q = ax1.quiver(0,0,0, 2,2,2, color='b')
    ax1.text(2/2+0.2, 2/2-0.5, 2/2+0.5, r'$\mathbf{v}$', fontsize=12)

    q = ax1.quiver(0,0,0, -0.5,4,1.5, color='r')
    ax1.text(-0.5/2-0.5, 4/2+0.5, 1.5/2-0.7, r'$\mathbf{u}$', fontsize=12)

    q = ax1.quiver(0,0,0, -5,-4,9, color='purple')
    ax1.text(-5/2+0.5, -4/2+1, 9/2-2, r'$\mathbf{v} \times \mathbf{u}$', fontsize=12)

    plt.show()