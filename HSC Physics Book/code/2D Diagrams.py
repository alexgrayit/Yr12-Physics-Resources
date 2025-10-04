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





xmin = -4
xmax = 4

xTickIncrement = 5


ymin = -4
ymax = 4

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
ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw, head_width=0, head_length=0, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw, head_width=yhw, head_length=yhl, overhang=ohg, length_includes_head=True, clip_on=False)

# clip_on = False if only positive x or y values.


ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')



#Set x and y labels
#ax.text(xmax + 0.1, 0.1, r'$x$', fontsize=12)
#ax.text(0.1, ymax+0.1, r'$y$', fontsize=12)


ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


xTicks = np.arange(ceil(xmin + xTickIncrement), xmax, xTickIncrement)
yTicks = np.arange(ymin + yTickIncrement, ymax, yTickIncrement)
yTicks = np.delete(yTicks, np.where(yTicks == 0))
xTicks = np.delete(xTicks, np.where(xTicks == 0))

ax.set_xticks(xTicks)
ax.set_yticks(yTicks)

ax.set_xticks([])
ax.set_yticks([])

def f(x):
    y = np.abs(x)
    return y


rScale = 1


N = 3
x = np.linspace(xmin, xmax, N)
y = f(x)

#ax.plot(x, y, color='black')

#matplotlib.pyplot.arrow(x, y, dx, dy, **kwargs)[source]
ax.arrow(xmin, f(xmin), -xmin, -f(xmin), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
ax.arrow(0, 0, xmax, f(xmax), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)



#class matplotlib.patches.Arc((x, y), width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs)
a = Arc((0, 0), 2, 2, 0, 135, 180, color='black', lw=1)
ax.add_patch(a)
ax.text(-1.4, 0.2, r'$\theta_i$', fontsize=12)

a = Arc((0, 0), 2, 2, 0, 0, 45, color='black', lw=1)
ax.add_patch(a)
ax.text(1.1, 0.2, r'$\theta_r$', fontsize=12)


plt.tick_params(left = False, bottom = False)
plt.show()