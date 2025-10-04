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




xmin = -0.1
xmax = 5

xTickIncrement = 1


ymin = -0.1
ymax = 5

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
ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)

ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw, head_width=yhw, head_length=yhl, overhang=ohg, length_includes_head=True, clip_on=False)

# clip_on = False if only positive x or y values.


ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_aspect('equal', adjustable='box')



#Set x and y labels
ax.text(xmax + 0.1, 0.1, r'$x$', fontsize=12)
ax.text(0.1, ymax+0.1, r'$y$', fontsize=12)


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
    y = (x-2)**3 - (x-2)/2
    return y

def f(x):
    shift = 0
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
ax.plot(x, y, color='black')


#holeCircle = plt.Circle((0, 2), 0.05, color='black', fill=False)
#ax.add_patch(holeCircle)

plotDerivative = True
if plotDerivative:
    # defining functions and variables for derivative diagrams
    x1 = 0.7
    dx = 1

    #x vertical line
    ax.plot(x1, f(x1) + 0.2*0.05, marker='o', color='black', markersize=4)
    ax.plot([x1, x1], [0, f(x1) + 0.2*0.05], marker='', linestyle='--', color='dimgrey', linewidth=1)
    #ax.text(x1 - 0.05, -0.4, r'$x$', fontsize=12, color='dimgrey')
    ax.text(x1 - 0.1, -0.25, r'$x$', fontsize=12)

    #dx right vertical line
    ax.plot(x1+dx, f(x1+dx) + 0.2*0.1, marker='o', color='black', markersize=4)
    ax.plot([x1+dx, x1+dx], [0, f(x1+dx) + 0.2*0.1], marker='', linestyle='--', color='black', linewidth=1)
    ax.text(x1+dx-0.1, -0.25, r'$x + h$', fontsize=12)

    #dx left vertical line
    #ax.plot(x1-dx, f(x1-dx) + 0.2*0.1, marker='o', color='black', markersize=4)
    #ax.plot([x1-dx, x1-dx], [0, f(x1-dx) + 0.2*0.1], marker='', linestyle='--', color='black', linewidth=1)
    #ax.text(x1-dx-0.4, -0.25, r'$x - h$', fontsize=12)


    #y horizontal line
    #ax.plot([0, x1], [f(x1) + 0.2*0.05, f(x1) + 0.2*0.05], marker='', linestyle='--', color='black', linewidth=1)
    #ax.text(0 - 0.5, f(x1) + 0.2*0.05 - 0.05, r'$f(x)$', fontsize=12)

    #dy top horizontal line
    ax.plot([0, x1+dx], [f(x1+dx) + 0.2*0.1, f(x1+dx) + 0.2*0.1], marker='', linestyle='--', color='black', linewidth=1)
    ax.text(0 - 1, (f(x1+dx) + 0.2*0.1) - 0.05, r'$f(x + h)$', fontsize=12)

    # dy bottom horizontal line (y)
    ax.plot([0, x1], [f(x1) + 0.2 * 0.1, f(x1) + 0.2 * 0.1], marker='', linestyle='--', color='black', linewidth=1)
    ax.text(0 - 0.7, (f(x1) + 0.2 * 0.1) - 0.05, r'$f(x)$', fontsize=12)

    #dy bottom horizontal line (y-dy)
    #ax.plot([0, x1-dx], [f(x1-dx) + 0.2*0.1, f(x1-dx) + 0.2*0.1], marker='', linestyle='--', color='black', linewidth=1)
    #ax.text(0 - 1, (f(x1-dx) + 0.2*0.1) - 0.05, r'$f(x - h)$', fontsize=12)



    #sloped secant x-dx to x+dx
    #ax.plot([0, 5], [df(0, x1-dx, x1+dx), df(5, x1-dx, x1+dx)], marker='', linestyle='-', color='dimgrey', linewidth=1)

    # sloped secant x to x+dx
    ax.plot([0, 5], [df(0, x1, x1+dx), df(5, x1, x1+dx)], marker='', linestyle='-', color='dimgrey', linewidth=1)

    #Delta X Gap Arrow and Label
    gap = 0.005
    height = 0.1
    #right dx
    plt.annotate(text='', xy=(x1 + gap, height), xytext=(x1+dx - gap + 0.01, height), arrowprops=dict(arrowstyle='<->'))
    xAvg = (x1+dx + x1) / 2
    ax.text(xAvg - 0.05, height + 0.1, r'$h$', fontsize=12)

    #left dx
    #plt.annotate(text='', xy=(x1 + gap, height), xytext=(x1-dx - gap + 0.01, height), arrowprops=dict(arrowstyle='<->'))
    #xAvg = (x1-dx + x1) / 2
    #ax.text(xAvg - 0.05, height + 0.1, r'$h$', fontsize=12)



#path length diagram
#x0 = 1.5
#x1 = 3

#y0 = f(x0)
#y1 = f(x1)

#plot points
#ax.plot(x0, y0 + 0.2*0.05, marker='o', color='black', markersize=4)
#ax.plot(x1, y1 + 0.2*0.05, marker='o', color='black', markersize=4)

#plot line between points
#ax.plot([x0, x1], [y0, y1], marker='', linestyle='-', color='dimgrey', linewidth=1)

#plot the horizontal and vertical projection lines
#ax.plot([x0, x1], [y0, y0], marker='', linestyle='--', color='black', linewidth=1)
#ax.plot([x1, x1], [y0, y1], marker='', linestyle='--', color='black', linewidth=1)

#ax.text((x0 + x1)/2 -0.05, y0-0.4, r'$dx$', fontsize=12)
#ax.text(x1+0.15, (y0+y1)/2 -0.2, r'$dy$', fontsize=12)
#ax.text((x0 + x1)/2 - 0.2, (y0+y1)/2 + 0.2, r'$dl$', fontsize=12)


#Circle Diagram
#circle = plt.Circle((0, 0), 1, color='black', fill=False, ls='--')
#ax.add_patch(circle)
theta0 = np.pi/4
theta1 = np.pi/2
radDegConvert = 180/np.pi

# r1 vector
# ax.arrow(0, 0, np.cos(theta0), np.sin(theta0), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
# ax.text(np.cos(theta0)/2, np.sin(theta0)/2-0.1, r'$\vec{\mathbf{r}}_1$', fontsize=12)
#
# r2 vector
# ax.arrow(0, 0, np.cos(theta1), np.sin(theta1), fc='k', ec='k', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
# ax.text(np.cos(theta1)/2-0.15, np.sin(theta1)/2, r'$\vec{\mathbf{r}}_2$', fontsize=12)
# 
# delta r vector
# drx = np.cos(theta1) - np.cos(theta0)
# dry = np.sin(theta1) - np.sin(theta0)
# ax.arrow(np.cos(theta0), np.sin(theta0), drx, dry, fc='black', ec='black', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
# ax.text(0.95*np.cos((theta0 + theta1)/2 + 3/radDegConvert), 0.95*np.sin((theta0 + theta1)/2 + 3/radDegConvert), r'$\delta \vec{\mathbf{r}}$', fontsize=12)

#vel 1 vector
#ax.arrow(np.cos(theta0), np.sin(theta0), 0.5*-np.sin(theta0), 0.5*np.cos(theta0), fc='purple', ec='purple', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.text(1.05*np.cos((theta0 + theta1)/2 - 8/radDegConvert), 1.05*np.sin((theta0 + theta1)/2 - 8/radDegConvert), r'$\vec{\mathbf{v}}_1$', fontsize=12, color='purple')

#vel 2 vector
#ax.arrow(np.cos(theta1), np.sin(theta1), 0.5*-np.sin(theta1), 0.5*np.cos(theta1), fc='purple', ec='purple', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.text(1.1*np.cos(theta1 + 15/radDegConvert), 1.1*np.sin(theta1 + 15/radDegConvert), r'$\vec{\mathbf{v}}_2$', fontsize=12, color='purple')


#delta theta for radius
#class matplotlib.patches.Arc((x, y), width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs)
#a = Arc((0, 0), 0.5, 0.5, 0, theta0*radDegConvert, theta1*radDegConvert, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(np.cos((theta0 + theta1)/2 + 10/radDegConvert)*0.27, np.sin((theta0 + theta1)/2 + 10/radDegConvert)*0.27, r'$\delta \theta$', fontsize=12)

#phi angles for radius
#a = Arc((np.cos(theta0), np.sin(theta0)), 0.4, 0.4, 0, 180 + np.arctan(dry/drx)*radDegConvert, 180 + theta0*radDegConvert, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(np.cos(theta0)- 0.3, np.sin(theta0)-0.1, r'$\phi$', fontsize=12)

#a = Arc((np.cos(theta1), np.sin(theta1)), 0.4, 0.4, 0, 180 + theta1*radDegConvert, np.arctan(dry/drx)*radDegConvert, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(np.cos(theta1) + 0.12, np.sin(theta1)-0.25, r'$\phi$', fontsize=12)


#vel 1 vector from origin
#ax.arrow(0, 0, 0.8*-np.sin(theta0), 0.8*np.cos(theta0), fc='purple', ec='purple', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.text(0.4*np.cos(theta0 + 90/radDegConvert), 0.4*np.sin(theta0 + 90/radDegConvert), r'$\vec{\mathbf{v}}_1$', fontsize=12, color='purple')

#vel 2 vector from origin
#ax.arrow(0, 0, 0.8*-np.sin(theta1), 0.8*np.cos(theta1), fc='purple', ec='purple', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.text(0.4*np.cos(theta1 + 110/radDegConvert)-0.05, 0.4*np.sin(theta1 + 110/radDegConvert), r'$\vec{\mathbf{v}}_2$', fontsize=12, color='purple')

#delta v
#dvx = 0.8*-np.sin(theta1) - 0.8*-np.sin(theta0)
#dvy = 0.8*np.cos(theta1) - 0.8*np.cos(theta0)
#ax.arrow(0.8*-np.sin(theta0), 0.8*np.cos(theta0), dvx, dvy, fc='purple', ec='purple', lw=lw, head_width=hw, head_length=hl, overhang=ohg, length_includes_head=True, clip_on=False)
#ax.text(np.cos((theta0 + theta1)/2 + 95/radDegConvert)*0.9, np.sin((theta0 + theta1)/2 + 95/radDegConvert)*0.9, r'$\delta\vec{\mathbf{v}}$', fontsize=12, color='purple')

#delta theta for velocity
#class matplotlib.patches.Arc((x, y), width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs)
#a = Arc((0, 0), 0.3, 0.3, 0, theta0*radDegConvert + 90, theta1*radDegConvert + 90, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(np.cos((theta0 + theta1)/2 + 105/radDegConvert)*0.3, np.sin((theta0 + theta1)/2 + 105/radDegConvert)*0.3, r'$\delta \theta$', fontsize=12)

#phi angles for velocity
#a = Arc((0.8*-np.sin(theta0), 0.8*np.cos(theta0)), 0.3, 0.3, 0, 180+np.arctan(dvy/dvx)*radDegConvert, 270 + theta0*radDegConvert, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(0.8*-np.sin(theta0), 0.8*np.cos(theta0) - 0.26, r'$\phi$', fontsize=12)

#a = Arc((0.8*-np.sin(theta1), 0.8*np.cos(theta1)), 0.3, 0.3, 0, 0, np.arctan(dvy/dvx)*radDegConvert, color='black', lw=1, ls='-')
#ax.add_patch(a)
#ax.text(0.8*-np.sin(theta1)+0.13, 0.8*np.cos(theta1)+0.09, r'$\phi$', fontsize=12)



if False:
    #Integration Setup
    trapezoid = True
    fullArea = False
    xValues = np.array([])
    delta = 1
    X = 1
    Xmax = 5
    while X <= Xmax:
        xValues = np.append(xValues, X)
        X += delta

    yValues = f(xValues)


    #Integration Lines
    count = 0
    for i in xValues:

        if(count == 0):
            if f(xValues[count]) >= 0:
                height = -0.3
            else:
                height = 0.2
            ax.text(xValues[count] - 0.05, height, r'$a$', fontsize=12)
            ax.plot([xValues[count], xValues[count]], [0, f(xValues[count])], marker='', linestyle='--', color='black',
                    linewidth=1)



        if (count + 1 != len(xValues)):
            if not fullArea:
                ax.plot(xValues[count], f(xValues[count]), marker='o', color='black', markersize=4)
                ax.plot([xValues[count], xValues[count]], [0, f(xValues[count])], marker='', linestyle='--', color='black', linewidth=1)

                gap = 0.005
                if f(xValues[count]) >= 0:
                    height = 0.1
                    dheight = 0.2
                else:
                    height = -0.1
                    dheight = -0.5
                plt.annotate(text='', xy=(xValues[count] + gap, height), xytext=(xValues[count + 1] - gap + 0.01, height),
                             arrowprops=dict(arrowstyle='<->'))
                xAvg = (xValues[count + 1] + xValues[count]) / 2
                ax.text(xAvg - 0.15, dheight, r'$\Delta x$', fontsize=12)

        elif(count != 0):
            if f(xValues[count]) >= 0:
                height = -0.3
            else:
                height = 0.1
            ax.text(xValues[count] - 0.05, height, r'$b$', fontsize=12)
            ax.plot([xValues[count], xValues[count]], [0, f(xValues[count])], marker='', linestyle='--', color='black',
                    linewidth=1)
            if not fullArea:
                ax.plot(xValues[count], f(xValues[count]), marker='o', color='black', markersize=4)
        count += 1


    #Integration Slopes
    count = 0
    for i in xValues[0:-1]:
        ax.plot([xValues[count], xValues[count+1]],
                [df(xValues[count], xValues[count], xValues[count+1]), df(xValues[count+1], xValues[count], xValues[count+1])],
                marker='', linestyle='-', color='black', linewidth=1)
        count += 1


    #Integration Areas
    if not fullArea:
        count = 0
        for i in xValues[0:-1]:
            xVals = [xValues[count], xValues[count], xValues[count + 1], xValues[count + 1]]
            yVals = [0, f(xValues[count]), f(xValues[count + 1]), 0]
            ax.fill(xVals, yVals, "grey", alpha=0.5)

            count += 1
    else:
        #shade whole area
        count = 0
        xVals = np.array([])
        yVals = np.array([])
        length = len(xValues)
        while count < 2*length:
            if count < length:
                #print(xValues[count])
                xVals = np.append(xVals, xValues[count])
                yVals = np.append(yVals, yValues[count])
            else:
                #print(xValues[-(count-length) - 1])
                xVals = np.append(xVals, xValues[-(count-length) - 1])
                yVals = np.append(yVals, 0)
            count+=1
        ax.fill(xVals, yVals, "grey", alpha=0.5)



#plt.tick_params(left = False, bottom = False)
plt.show()