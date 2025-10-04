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

xmin = -0.3
xmax = 50

xTickIncrement = 2


ymin = -30
ymax = 5

yTickIncrement = 2


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)



ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')



#Set x and y labels
ax.text(xmax + 0.3, 0.3, r'$r$', fontsize=12)
ax.text(20, 1, r'$(m \times 10^{6})$', fontsize=12)
ax.text(-0.3, ymax+0.5, r'$U_g$', fontsize=12)
ax.text(-6, -10, r'$(J \times 10^{9})$', fontsize=12, rotation='vertical')

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


xTicks = np.arange(ceil(xmin + xTickIncrement), xmax, xTickIncrement)
yTicks = np.arange(ymin + yTickIncrement, ymax, yTickIncrement)
#yTicks = np.delete(yTicks, np.where(yTicks == 0))
xTicks = np.delete(xTicks, np.where(xTicks == 0))

ax.set_xticks(xTicks)
ax.set_yticks(yTicks)

ax.arrow(0.02, ymax-0.2, 0, 0.01, head_width=0.3, head_length=0.2, fc='k', ec='k')
ax.arrow(xmax-0.2, -0.01, 0.01, 0, head_width=0.3, head_length=0.2, fc='k', ec='k')
#ax.set_xticklabels([])
#ax.set_yticklabels([])

GMm = 3*(10**16) * (10**(-9))
rScale = (10**6)
rE = 6.371*(10**6)/rScale


step = 0.1
rbefore = np.arange(0+step, rE, step)
rafter = np.arange(rE, xmax, step)

ax.plot(rbefore, -GMm/(rbefore*rScale), color='gray')
ax.plot(rafter, -GMm/(rafter*rScale))



ax.axvline(x=rE, color='gray', alpha=0.5, linestyle='dashed')
ax.text(rE -0.5, ymin + -1.5, r'$r_E$', fontsize=12)


plt.show()