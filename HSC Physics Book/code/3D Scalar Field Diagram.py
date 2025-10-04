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
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 12)


xmin = -30
xmax = 30

xTickIncrement = 2


ymin = -30
ymax = 30

yTickIncrement = 2

zmin = -23
zmax = 23

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
    ax1 = fig.gca(projection='3d')
    ax1.grid(False)
    ax1._axis3don = False
    ax1.set_aspect(aspect='auto')

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
    ax1.text(0.3, 0.3, zmax + 0.5, r'$z$', fontsize=12)

    #N = 50
    #stride = 2
    #u = np.linspace(0, 2 * np.pi, N)
    #v = np.linspace(0, np.pi, N)
    #x = (rE/rScale)*np.outer(np.cos(u), np.sin(v))
    #y = (rE/rScale)*np.outer(np.sin(u), np.sin(v))
    #z = (rE/rScale)*np.outer(np.ones(np.size(u)), np.cos(v))
    #ax1.plot_surface(x, y, z, color='royalblue', linewidth=0.0, cstride=stride, rstride=stride, alpha=0.8, shade=False)


    step = 5
    xp = np.linspace(xmin+5, xmax-5, round(float(xmax - xmin - 10)/step))
    yp = np.linspace(ymin+5, ymax-5, round(float(ymax - ymin - 10) / step))

    Xp, Yp = np.meshgrid(xp, yp)
    U = 1/(GPE(Xp, Yp))
    #U = np.where(Xp ** 2 + Yp ** 2 >= (rE / rScale - 1) ** 2, U, GPE(rE / rScale, 0))

    Xp = np.where(Xp ** 2 + Yp ** 2 <= ((xmax + ymax)/2 - 5) ** 2, Xp, np.nan)
    #Xp = np.where(Xp ** 2 + Yp ** 2 >= (rE/rScale - 2) ** 2, Xp, np.nan)
    Yp = np.where(Xp ** 2 + Yp ** 2 <= ((xmax + ymax) / 2 - 5) ** 2, Yp, np.nan)
    #Yp = np.where(Xp ** 2 + Yp ** 2 >= (rE / rScale - 2) ** 2, Yp, np.nan)

    zz = Xp + Yp


    color_dimension = U  # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(cmap='inferno')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    Zp = 0*zz

    #ax1.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.5, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

    for i in xp:
        for j in yp:
            ax1.text(i, j, 0, r'${Potential: .2f}$'.format(Potential = GPE(i, j)), fontsize=6)


    plt.show()