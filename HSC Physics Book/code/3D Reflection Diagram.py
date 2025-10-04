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
from mpl_toolkits import mplot3d

xmin = -20
xmax = 20

xTickIncrement = 2


ymin = -20
ymax = 20

yTickIncrement = 2

zmin = -17
zmax = 17

zTickIncrement = 2




GMm = 3 * (10 ** 16) * (10 ** (-9))
rE = 6.371 * (10**6)
rScale = (10 ** 1)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)




if __name__ == '__main__':

    center = [0, 0, 0]


    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.grid(False)
    ax1._axis3don = False

    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylabel(r'$y$', fontsize=12)
    ax1.set_ylim(ymin, ymax)
    ax1.set_zlabel(r'$z$', fontsize=12)
    ax1.set_zlim(zmin, zmax)

    # Here we create the arrows:
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)


    axisThick = 1
    #primary axis arrows
    a = Arrow3D([0, xmax], [0, 0], [0, 0], **arrow_prop_dict, color='black', lw=axisThick)
    #ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, ymax], [0, 0], **arrow_prop_dict, color='black', lw=axisThick)
    #ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, zmax], **arrow_prop_dict, color='black', lw=axisThick)
    ax1.add_artist(a)

    #secondary axis arrows
    a = Arrow3D([0, xmin], [0, 0], [0, 0], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    #ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, ymin], [0, 0], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    #ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, zmin], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    ax1.add_artist(a)

    #Axes labels
    #ax1.text(xmax + 0.5, 0.3, 0.5, r'$x$', fontsize=12)
    #ax1.text(-0.3, ymax + 0.7, 0.5, r'$y$', fontsize=12)
    ax1.text(0.3, 0.3, zmax + 0.5, r'$A$', fontsize=12)

    N = 201
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    X, Y, Z = np.meshgrid(x, y, z)

    p = 5*np.sin(np.sqrt(x**2 + y**2))
    p = np.where(x >= 0, p, np.nan)

    ax1.plot(x, y, p, color='blue')
    ax1.plot(-x, y, -p, color='royalblue')
    a = Arrow3D([0, xmax], [0, ymax], [0, 0], **arrow_prop_dict, color='black', lw=axisThick)
    ax1.add_artist(a)
    a = Arrow3D([xmin, 0], [ymax, 0], [0, 0], **arrow_prop_dict, color='black', lw=axisThick)
    ax1.add_artist(a)

    Wallxs, Wallzs = np.meshgrid(np.array([xmin, xmax]), np.array([zmin+13, zmax-13]))
    Wallys = Wallxs*0
    ax1.plot_surface(Wallxs, Wallys, Wallzs, color='black', alpha=0.3)
    plt.show()