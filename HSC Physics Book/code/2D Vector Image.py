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

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.patches import Arc

xmin = -10
xmax = 10

xTickIncrement = 2


ymin = -10
ymax = 10

yTickIncrement = 2


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)



ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')


#Dot product diagram
#ax.quiver(0, 0, 4, 6, color='blue', units='xy', scale=1)
#ax.text(4/2 - 0.5, 6/2 + 0.3, r'$\mathbf{v}$', fontsize=12)

#ax.quiver(0, 0, 6, -8, color='blue', units='xy', scale=1)
#ax.text(6/2 - 0.7, -8/2 - 0.5, r'$\mathbf{u}$', fontsize=12)


#ax.quiver(6, -8, 4, 6, color='blue', units='xy', scale=1, alpha=0.7)
#ax.text(4/2 - 0.5 + 6.7, 6/2 + 0.3 - 8.7, r'$\mathbf{v}$', fontsize=12)

#ax.quiver(4, 6, 6, -8, color='blue', units='xy', scale=1, alpha=0.7)
#ax.text(6/2 - 0.7 + 4.7, -8/2 - 0.5 + 6.7, r'$\mathbf{u}$', fontsize=12)



#Triangle
#shapeX = [0, 4, 6]
#shapeY = [0, 6, -8]


#Parallelogram
#shapeX = [0, 4, 10, 6]
#shapeY = [0, 6, -2, -8]
#ax.fill(shapeX, shapeY, "b", alpha=0.5)

#ax.text(3.5, -3, r'$A = \left\|\mathbf{v} \times \mathbf{u}\right\|$', fontsize=12)


#Vector addition diagram
ax.quiver(0, 0, -6, 2, color='blue', units='xy', scale=1, alpha=1)
ax.text(-6/2, 2/2 + 0.3, r'$\mathbf{a}$', fontsize=12)

ax.quiver(0, 0, -2, 8, color='blue', units='xy', scale=1, alpha=1)
ax.text(-2/2 - 1.2, 8/2, r'$\mathbf{b}$', fontsize=12)

ax.quiver(-6, 2, 2, -8, color='blue', units='xy', scale=1, alpha=0.7)
ax.text(-6.5, -2.5, r'$\mathbf{-b}$', fontsize=12)

ax.quiver(0, 0, -4, -6, color='slateblue', units='xy', scale=1, alpha=1)
ax.text(-3, -5.3, r'$\mathbf{a} - \mathbf{b}$', fontsize=12)


#Point Vector Diagram
#ax.quiver(0, 0, -6, 2, color='blue', units='xy', scale=1, alpha=1)
#ax.text(-6/2 - 0.7, 2/2 + 0.5, r'$\mathbf{a}$', fontsize=12)
#ax.plot(-5, 6, 'bo', markersize=3)
#ax.text(-5 - 0.5, 6 + 1.5, r'$A$', fontsize=12)

#ax.quiver(0, 0, 4, 3, color='blue', units='xy', scale=1, alpha=1)
#ax.text(4/2, 3/2 - 1, r'$\mathbf{b}$', fontsize=12)
#ax.plot(6, 1, 'bo', markersize=3)
#ax.text(6 + 0.2, 1 + 0.5, r'$B$', fontsize=12)

#ax.quiver(0, 0, 11, -5, color='royalblue', units='xy', scale=1, alpha=1)
#ax.text(11/2 - 0.7, -5/2 - 1.5, r'$\vec{AB}$', fontsize=12)


#Set x and y labels
ax.text(xmax + 0.3, -0.3, r'$x$', fontsize=12)
ax.text(0.3, ymax+0.3, r'$y$', fontsize=12)

#class matplotlib.patches.Arc((x, y), width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs)
#a = Arc((0, 0), 2, 2, 0, -36.87 - 15, 33.69 + 20, color='black', lw=1)
#ax.add_patch(a)
#ax.text(1.3, 0.2, r'$\theta$', fontsize=12)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


xTicks = np.arange(ymin + yTickIncrement, ymax, yTickIncrement)
yTicks = np.arange(xmin + xTickIncrement, xmax, xTickIncrement)
yTicks = np.delete(yTicks, np.where(yTicks == 0))
xTicks = np.delete(xTicks, np.where(xTicks == 0))

ax.set_xticks(xTicks)
ax.set_yticks(yTicks)

ax.arrow(0.02, ymax-0.2, 0, 0.01, head_width=0.3, head_length=0.2, fc='k', ec='k')
ax.arrow(xmax-0.2, -0.01, 0.01, 0, head_width=0.3, head_length=0.2, fc='k', ec='k')
#ax.set_xticklabels([])
#ax.set_yticklabels([])

#plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.show()