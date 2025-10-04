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
from math import floor
from math import ceil

import numpy as np
from numpy import pi
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
from matplotlib.patches import Arc




xmin = 0
xmax = 4

xTickIncrement = 1


ymin = 0
ymax = 4

yTickIncrement = 1


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
yTicks = np.arange(np.round(ymin) + yTickIncrement, ymax, yTickIncrement)
yTicks = np.delete(yTicks, np.where(yTicks == 0))
xTicks = np.delete(xTicks, np.where(xTicks == 0))

#ax.set_xticks(xTicks)
#ax.set_yticks(yTicks)
#plt.setp( ax.yaxis.get_majorticklabels(), rotation=0, ha="left")

#ax.set_xticks(xTicks)
#ax.set_yticks(yTicks)

def func(x):
    y = (x)**3 - (x)/2
    return y

def f(x):
    shift = 2
    scale = 0.2
    y = scale * (func(x-shift) - func(-shift))
    #the extra -func(-shift) fixes the graph to the origin
    return y

#equation of the tangent line used in derivative limit diagram
def df(x, a, b):
    y = ((x-a)/(b-a))*(f(b)-f(a)) + f(a)
    return y


rScale = 1


N = 101
#x = np.linspace(0, xmax, N)
#y = f(x)

x = np.linspace(xmin, xmax, N)
y = f(x)
ax.plot(x, y, color='black', lw=0.5)

arrow1x = 1
dx = 0.001
horizLen = 0.2
#x, y, xdir, ydir
ax.arrow(arrow1x, f(arrow1x), horizLen, df(arrow1x + horizLen, arrow1x, arrow1x+dx)-f(arrow1x), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
ax.arrow(arrow1x, f(arrow1x), horizLen, df(arrow1x + horizLen - 0.2, arrow1x, arrow1x+dx) + 1 - f(arrow1x), fc='blue', ec='blue', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)

ax.text(arrow1x-0.1, f(arrow1x)+0.5, r'$\vec{F}$', fontsize=12, c='blue')
ax.text(arrow1x, f(arrow1x)-0.15, r'$d\vec{s}$', fontsize=12, c='k')

arrow1x = 2
ax.arrow(arrow1x, f(arrow1x), horizLen, df(arrow1x + horizLen, arrow1x, arrow1x+dx)-f(arrow1x), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
ax.arrow(arrow1x, f(arrow1x), -horizLen, df(arrow1x + horizLen - 0.2, arrow1x, arrow1x+dx) -0.5 - f(arrow1x), fc='blue', ec='blue', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)

ax.text(arrow1x-0.1, f(arrow1x)-0.5, r'$\vec{F}$', fontsize=12, c='blue')
ax.text(arrow1x, f(arrow1x)-0.2, r'$d\vec{s}$', fontsize=12, c='k')


plt.show()