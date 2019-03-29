# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:18:00 2019

@author: e0008730
"""

import types
import numpy as np


def PlotFunc3D( func:types.FunctionType, 
               xmin=0., xmax=1., xn=101, 
               ymin=0., ymax=1., yn=101 ):
    #lib for plotting
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #calculate data    
    xcoord = np.linspace( xmin, xmax, num=xn )
    ycoord = np.linspace( ymin, ymax, num=yn )    
    xs, ys = np.meshgrid( xcoord, ycoord )
    zs = func(xs, ys)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    pass
    
    
if __name__ == "__main__":
    def vol(T,K):
        invK = 10.
        invV = 0.20
        cvx = np.log(invK) - np.log(K)
        cvx_coef = np.exp(-T/0.25)
        slope = -0.02 - 0.03*np.exp(-T/0.25)
        return invV+slope*(K-invK)+cvx_coef*cvx
    
    def vol2(K,T):
        return vol(T,K)
    
    PlotFunc3D( vol2, xmin=5, xmax=15 )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    