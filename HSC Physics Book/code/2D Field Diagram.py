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

zmin = -30
zmax = 15

zTickIncrement = 2




GMm = 3 * (10 ** 16) * (10 ** (-9))
rE = 6.371 * (10**6)
rScale = (10 ** 6)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def GPE(x, y):

    r = np.sqrt(x ** 2 + y ** 2)

    return -GMm / (r * rScale)




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
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, ymax], [0, 0], **arrow_prop_dict, color='black', lw=axisThick)
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, zmax], **arrow_prop_dict, color='black', lw=axisThick)
    ax1.add_artist(a)

    #secondary axis arrows
    a = Arrow3D([0, xmin], [0, 0], [0, 0], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, ymin], [0, 0], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    ax1.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, zmin], **arrow_prop_dict, color='black', alpha=0.7, lw=axisThick)
    ax1.add_artist(a)

    # Give them a name:
    #ax1.text(0.0, 0.0, -0.2, r'$0$')
    ax1.text(xmax + 0.5, 0.3, 0.5, r'$x$', fontsize=12)
    ax1.text(-0.3, ymax + 0.7, 0.5, r'$y$', fontsize=12)
    ax1.text(0.3, 0.3, zmax + 0.5, r'$U_g$', fontsize=12)

    step = 0.35
    x = np.linspace(xmin, xmax, round(float(xmax - xmin - 20) / step))
    x = np.delete(x, np.where(x == 0))
    y = x

    Xb, Yb = np.meshgrid(x, y)
    Zb = GPE(Xb, Yb)

    Za = np.where(Xb ** 2 + Yb ** 2 + 1 >= (rE / rScale) ** 2, Zb, np.nan)
    Za = np.where(Xb ** 2 + Yb ** 2 < (xmax-5) ** 2, Za, np.nan)

    Zb = np.where(Xb**2 + Yb**2 <= (rE/rScale + 1)**2, Zb, np.nan)
    Zb = np.where(Xb ** 2 + Yb ** 2 > (0.5) ** 2, Zb, np.nan)

    ax1.plot_surface(Xb, Yb, Zb, rstride=1, cstride=1, color='grey', edgecolor='none', linewidth=0)
    ax1.plot_surface(Xb, Yb, Za, rstride=1, cstride=1, edgecolor='none', linewidth=0)


    # Plot a sin curve using the x and y axes.
    u = np.linspace(0, 2 * np.pi, 50)  # divide the circle into 50 equal parts
    h = np.linspace(zmin - 5, zmax - 10, 20)  # divide the height 1 into 20 parts
    x = np.outer((rE/rScale)*np.sin(u), np.ones(len(h)))  # x value repeated 20 times
    y = np.outer((rE/rScale)*np.cos(u), np.ones(len(h)))  # y value repeated 20 times
    z = np.outer(np.ones(len(u)), h)  # x,y corresponding height

    # Plot the surface
    #ax1.plot_surface(x, y, z, color='gray', alpha=1)

    plt.show()