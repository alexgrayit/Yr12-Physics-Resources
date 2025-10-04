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

#matplotlib.use('TkAgg') #for displaying in seperate window
matplotlib.use('Agg') #for gifs

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib import animation, rcParams
from PIL import Image

xmin = 0
xmax = 20

xTickIncrement = 2


ymin = -20
ymax = 20

yTickIncrement = 2

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
ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)

ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw, head_width=yhw, head_length=yhl, overhang=ohg, length_includes_head=True, clip_on=False)

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

plt.tick_params(left = False, bottom = False)

ax.set_xticklabels([])
ax.set_yticklabels([])
rScale = 1



waveSpeed = 5

def wavePos(t):
    tmax = 2
    if t <= tmax:
        xStart = xmin
    else:
        xStart = (t-tmax)*waveSpeed

    xFinish = t*waveSpeed
    if xFinish >= xmax-1:
        xFinish = xmax-1


    return xStart, xFinish


yList = np.array([])

N = 5000
f = 1
k = (2*pi)/(waveSpeed/f)
w = (2*pi)*(f)
A = 10

x = np.linspace(xmin, xmax, N)

curve, = ax.plot(x, 0*x, color='black')


time = 0
timeLim = 10



def update(i):
    t = (1/30)*i
    xStart, xFinish = wavePos(t)
    Y = np.piecewise(x, [x < xStart, ((x >= xStart) & (x <= xFinish)), x > xFinish], [np.nan, lambda x: A*np.sin(w*t - k*x), np.nan])

    curve.set_xdata(x)
    curve.set_ydata(Y)
    #ax.plot(x, Y, color='black')


anim = FuncAnimation(fig, update, frames=timeLim*30, interval=(1/30))

anim.save('sine_wave.gif', writer='imagemagick', fps=30)
#plt.show()