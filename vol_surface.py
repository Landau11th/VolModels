   
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
    from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    #calculate data    
    xcoord = np.linspace( xmin, xmax, num=xn )
    ycoord = np.linspace( ymin, ymax, num=yn )    
    xs, ys = np.meshgrid( xcoord, ycoord )
    zs = func(xs, ys)
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    pass
    
    
if __name__ == "__main__":
    def vol(T,K):
        v_inf = 0.1
        K_cnt = 10.
        v_cnt = .25
        slope_cnt = -0.05
        slope_decay_period = 3.
        coef = (v_cnt-v_inf)
        return ((v_inf + coef*np.exp( (K-K_cnt)*slope_cnt/coef )) - v_cnt )*np.exp(-T/slope_decay_period) + v_cnt
        

#        #unesirable negative vol 
#        invK = 10.
#        invV = 0.20
#        cvx = np.log(invK) - np.log(K)
#        cvx_coef = np.exp(-T/0.25)
#        slope = -0.02 - 0.03*np.exp(-T/0.25)
#        return invV+slope*(K-invK)+cvx_coef*cvx
    
    def vol2(K,T):
        return vol(T,K)
    
    PlotFunc3D( vol2, xmin=8, xmax=12, ymin=0., ymax=5. )
