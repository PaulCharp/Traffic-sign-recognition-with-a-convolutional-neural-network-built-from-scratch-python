# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 23:30:56 2019

@author: paulc
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #Allow me to create 3D plot 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import time
import numpy as np
from PIL import Image

############################################3D plot configuration####################################################################################
#From https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
#Here we'll plot in 3D using a color warm scale
def colorWarm3DPlot(X,Y,Z,XLabel,YLabel,ZLabel, Title, x_scale = 'linear', y_scale = 'linear', z_scale = 'linear'):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    #Shape the data 
    #X, Y = np.meshgrid(X, Y)
    print(np.shape(X))
    print(np.shape(Y))
    print(np.shape(Z))
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0)
    
    #Config the axes (when playing with learning rates, it' easier to have logarithmic scales)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_zscale(z_scale)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #Add titles and legends
    ax.set_xlabel(XLabel)
    ax.set_ylabel(YLabel)
    ax.set_zlabel(ZLabel)
    plt.title(Title)
    
    plt.show()
    


def wire3DPlot(XY,Z,XLabel, YLabel,Title, ZLabel, x_scale = 'linear', y_scale = 'linear', z_scale = 'linear'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #Shape the data
    X,Y = XY , XY.transpose()
    ax.plot_wireframe(X, Y, Z, cmap = cm.coolwarm, rstride=10, cstride=10)
    #Config the axes (when playing with learning rates, it' easier to have logarithmic scales)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_zscale(z_scale)
    
    #Add titles and legends
    ax.set_xlabel(XLabel)
    ax.set_ylabel(YLabel)
    ax.set_zlabel(ZLabel)
    plt.title(Title)
    plt.show()