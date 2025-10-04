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

    #Axes labels
    ax1.text(xmax + 0.5, 0.3, 0.5, r'$x$', fontsize=12)
    ax1.text(-0.3, ymax + 0.7, 0.5, r'$y$', fontsize=12)
    ax1.text(0.3, 0.3, zmax + 0.5, r'$z$', fontsize=12)


    N = 4
    x = np.linspace(xmin + 7, xmax - 7, N)
    y = np.linspace(ymin + 7, ymax - 7, N)
    #z = np.linspace(zmin + 7, zmax - 7, 3)

    z = np.linspace(0, 0, 1)

    print(0 in x)
    print(0 in y)
    print(0 in z)

    theta = np.linspace(0, 2, 16)
    theta = theta*pi
    phi = np.linspace(0, 2*pi, 10)
    rad = np.linspace(3, xmax-7, 10)


    #x = xmax * np.cos(theta)
    #y = xmax * np.sin(theta)


    #Q, P, r = np.meshgrid(theta, phi, rad)

    #X = r * np.sin(P) * np.cos(Q)
    #Y = r * np.sin(P) * np.sin(Q)
    #Z = r * np.cos(P)

    #X, Y, Z = np.meshgrid(x, y, z)
    R, Q, Z = np.meshgrid(rad, theta, z)
    X = R * np.cos(Q)
    Y = R * np.sin(Q)

    X = X*rScale
    Y = Y*rScale
    Z = Z*rScale



    k = 3*(10**1)

    # Curling Field
    Vx = k*(-Y / np.sqrt(X ** 2 + Y ** 2)**2)
    Vy = k*(X / np.sqrt(X ** 2 + Y ** 2)**2)
    Vz = 0*Z

    Cx = 0*X
    Cy = 0*Y
    Cz = k**1.7/((X**2 + Y**2)**(3/4))

    #Divergent Field
    #Vx = k*X/(np.sqrt(X**2 + Y**2 + Z**2)**1.5)
    #Vy = k*Y/(np.sqrt(X**2 + Y**2 + Z**2)**1.5)
    #Vz = k*Z/(np.sqrt(X**2 + Y**2 + Z**2)**1.5)

    # Voltage / Electric Field
    #k=-5*(10**2)
    #Vx = k * X / (X ** 2 + Y ** 2 + Z ** 2)
    #Vy = k * Y / (X ** 2 + Y ** 2 + Z ** 2)
    #Vz = k * Z / (X ** 2 + Y ** 2 + Z ** 2)

    ax1.quiver(X/rScale, Y/rScale, Z/rScale, Vx, Vy, Vz, color='firebrick', pivot='middle', linewidth=2, alpha=1)
    ax1.quiver(X/rScale, Y/rScale, Z/rScale, Cx, Cy, Cz, color='darkcyan', pivot='tail', alpha=0.5, linewidth=2)

    #N = 50
    #stride = 2
    #u = np.linspace(0, 2 * np.pi, N)
    #v = np.linspace(0, np.pi, N)
    #x = 0.5 * np.outer(np.cos(u), np.sin(v))
    #y = 0.5 * np.outer(np.sin(u), np.sin(v))
    #z = 0.6 * np.outer(np.ones(np.size(u)), np.cos(v))
    #ax1.plot_surface(x, y, z, color='blue', linewidth=0.0, cstride=stride, rstride=stride, alpha=0.8, shade=False)

#    coordsDone = []
#    for i in X:
#        for i1 in i:
#            for i2 in i1:
#                for j in Y:
#                    for j1 in j:
#                        for j2 in j1:
#                            for k in Z:
#                                for k1 in k:
#                                    for k2 in k1:
#                                        if not ([i2, j2, k2] in coordsDone):
#                                            #div = 3/(2*((i2**2 + j2**2 + k2**2) ** (3 / 4)))
#                                            #k = 1*(10**2)
#                                            #Voltage = k/np.sqrt(i2**2 + j2**2 + k2**2)
#                                            #ax1.text(i2/rScale - 0.2, j2/rScale - 0.2, k2/rScale - 0.2, r'${Div: .2f}$'.format(Div=div*1000), fontsize=8)
#                                            #ax1.text(i2/rScale - 0.2, j2/rScale - 0.2, k2/rScale - 0.2, r'${V: .2f}$'.format(V=Voltage), fontsize = 8, alpha=0.7)
#                                            coordsDone.append([i2, j2, k2])


    plt.show()