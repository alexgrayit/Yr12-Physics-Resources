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

xmin = -8
xmax = 8

xTickIncrement = 5


ymin = -8
ymax = 8

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
ax.text(xmax + 0.3, 0, r'$x$', fontsize=12)
ax.text(0, ymax+0.5, r'$y$', fontsize=12)


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

N = 10
x = np.linspace(xmin, xmax, N)
y = np.linspace(ymin, ymax, N)
print(0 in x)

theta = np.linspace(0, 2*pi, 8)
rad = np.linspace(1, xmax, 4)

Q, r = np.meshgrid(theta, rad)


X, Y = np.meshgrid(x, y)
#X = r*np.cos(Q)*rScale
#Y = r*np.sin(Q)*rScale

#X = np.where(X ** 2 + Y ** 2 >= (1) ** 2, X, np.nan)
#Y = np.where(X ** 2 + Y ** 2 >= (1) ** 2, Y, np.nan)

#k = 0.5
#Vx = k*(-Y/(np.sqrt(X**2 + Y**2)**1.5))
#Vy = k*(X/(np.sqrt(X**2 + Y**2)**1.5))

div = 1/(2*((X**2 + Y**2)**(3/4)))


coordsDone = []
for i in X:
    for n in i:
        for j in Y:
            for l in j:
                if not([n, l] in coordsDone):
                    ax.text(n-0.2, l-0.2, r'${Div: .2f}$'.format(Div=1/(2*((n**2 + l**2)**(3/4)))), fontsize=8)
                    coordsDone.append([n, l])


#theta = np.linspace(0, 2*np.pi, 12)
#rad = np.linspace(1, xmax, 4)

#Q, r = np.meshgrid(theta, rad)

#X = r*np.cos(Q)*rScale
#Y = r*np.sin(Q)*rScale

k = 2
#Vx = k*((X+10*rScale)/(np.sqrt((X+10*rScale)**2 + Y**2)**3)) - k*((X-10*rScale)/(np.sqrt((X-10*rScale)**2 + Y**2)**3))
#Vy = k*(Y/(np.sqrt((X+10*rScale)**2 + Y**2)**3)) - k*(Y/(np.sqrt((X-10*rScale)**2 + Y**2)**3))

Vx = k*((X)/(np.sqrt((X)**2 + Y**2))**1.5)
Vy = k*(Y/(np.sqrt((X)**2 + Y**2))**1.5)


#circle1 = plt.Circle((0, 0), 1, color='b')
#circle2 = plt.Circle((0, 0), 1, color='r')
#ax.add_patch(circle1)
#ax.add_patch(circle2)

ax.quiver(X, Y, Vx, Vy, color='royalblue', units='xy', scale=1, pivot='middle', alpha=0.75)



plt.show()