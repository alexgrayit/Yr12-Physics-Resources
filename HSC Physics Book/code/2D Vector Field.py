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
from math import ceil

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.patches import Arc

xmin = -20
xmax = 20

xTickIncrement = 5


ymin = -20
ymax = 20

yTickIncrement = 5


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#AXIS ARROWS
# removing the default axis on all sides:
for side in ['bottom', 'right', 'top', 'left']:
    ax.spines[side].set_visible(False)

# removing the axis ticks
plt.xticks([])  # labels
plt.yticks([])
ax.xaxis.set_ticks_position('none')  # tick markers
ax.yaxis.set_ticks_position('none')

# wider figure for demonstration
#fig.set_size_inches(4, 2.2)

# get width and height of axes object to compute
# matching arrowhead length and width
dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1. / 40. * (ymax - ymin)
hl = 1. / 40. * (xmax - xmin)
lw = 0.5  # axis line width
ohg = 0.3  # arrow overhang

# compute matching arrowhead length and width
yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

# draw x and y axis
ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
         head_width=hw, head_length=hl, overhang=ohg,
         length_includes_head=True, clip_on=False)

ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
         head_width=yhw, head_length=yhl, overhang=ohg,
         length_includes_head=True, clip_on=False)

# clip_on = False if only positive x or y values.



ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')



#Set x and y labels
ax.text(xmax + 0.3, 0.3, r'$x$', fontsize=12)
ax.text(0.3, ymax+0.5, r'$y$', fontsize=12)


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


xTicks = np.arange(ceil(xmin + xTickIncrement), xmax, xTickIncrement)
yTicks = np.arange(ymin + yTickIncrement, ymax, yTickIncrement)
yTicks = np.delete(yTicks, np.where(yTicks == 0))
xTicks = np.delete(xTicks, np.where(xTicks == 0))

ax.set_xticks(xTicks)
ax.set_yticks(yTicks)



ax.set_xticklabels([])
ax.set_yticklabels([])
rScale = 1


N = 12
x = np.linspace(xmin+1.5, xmax-1.5, N)
y = np.linspace(ymin+1, ymax-1, N)


theta = np.linspace(0, 2*pi, 24)
rad = np.linspace(3, xmax-2, 10)

Q, r = np.meshgrid(theta, rad)


X, Y = np.meshgrid(x, y)

#X = r*np.cos(Q)*rScale
#Y = r*np.sin(Q)*rScale


#Curling Graph
#k = 1.5
#Vx = k*(-Y/(np.sqrt(X**2 + Y**2)**1.5))
#Vy = k*(X/(np.sqrt(X**2 + Y**2)**1.5))

#Single Charge
#k = 5
#Vx = k*(X/(np.sqrt(X**2 + Y**2))**1.5)
#Vy = k*(Y/(np.sqrt(X**2 + Y**2))**1.5)

#Two charges Example
#k = 50
#Vx = k*((X+10*rScale)/(np.sqrt((X+10*rScale)**2 + Y**2)**3)) - k*((X-10*rScale)/(np.sqrt((X-10*rScale)**2 + Y**2)**3))
#Vy = k*(Y/(np.sqrt((X+10*rScale)**2 + Y**2)**3)) - k*(Y/(np.sqrt((X-10*rScale)**2 + Y**2)**3))

d = 3
#Double Charge
#Vx = np.where(((X-10)**2 + Y**2) > d**2, Vx, np.nan)
#Vx = np.where(((X+10)**2 + Y**2) > d**2, Vx, np.nan)
#Vy = np.where(((X-10)**2 + Y**2) > d**2, Vy, np.nan)
#Vy = np.where(((X+10)**2 + Y**2) > d**2, Vy, np.nan)

#Single Charge
#Vx = np.where((X**2 + Y**2) >= d**2, Vx, np.nan)
#Vy = np.where((X**2 + Y**2) >= d**2, Vy, np.nan)

#Single Charge
#circle1 = plt.Circle((0, 0), 2, color='b')
#ax.add_patch(circle1)

#Pair of Charges
#circle1 = plt.Circle((10, 0), 2, color='b')
#circle2 = plt.Circle((-10, 0), 2, color='r')
#ax.add_patch(circle1)
#ax.add_patch(circle2)


#Weird Curl Example
k = 0.15
Vx = k*Y
Vy = 0*X

ax.quiver(X, Y, Vx, Vy, color='rebeccapurple', units='xy', scale=1, pivot='middle')


plt.tick_params(left = False, bottom = False)
plt.show()