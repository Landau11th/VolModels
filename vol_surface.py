   
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:18:00 2019
@author: e0008730
"""

import types
import numpy as np


def PlotFunc3D( func:types.FunctionType, 
               xmin=0., xmax=1., xn=101, 
               ymin=0., ymax=1., yn=101,
               labels=["","",""]):
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
    #ax.set_zlim(-5, 5)
    #ax = fig.gca(projection='3d')
    # Plot the surface.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    pass

ATM_VOL = 0.25
#example
def vol(T,K):
    v_inf = 0.1
    K_cnt = 1.
    v_cnt = ATM_VOL
    slope_cnt = -0.25
    coef = (v_cnt-v_inf)
    
    slope_decay_period = 4.
    temp = T/slope_decay_period
    #time_decay = np.exp(-temp)
    #time_decay = np.exp(-temp) + 0.5*(1 - np.exp(-temp))
    time_decay = temp*temp/2. - temp + 1.
    return ((v_inf + coef*np.exp( (K-K_cnt)*slope_cnt/coef )) - v_cnt )*time_decay + v_cnt
    
from scipy.stats import norm    
def BS_price( S, K, T, r, sigma ):
    d1 = np.log( S/K ) + (r + sigma*sigma/2.)*T
    d1 = d1/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BS_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))    
    vega = S * norm.cdf(d1) * np.sqrt(T)
    return vega

def BS_call_implied_vol(S, K, T, r, C, sigma_init=0.15, tol = 0.00001):
    res = sigma_init
    price = BS_price( S, K, T, r, res )
        
    while abs(price - C) > tol:
        res -= (price-C)/BS_vega(S,K,T,r,res)
        price = BS_price( S, K, T, r, res )
    return res

def BS_theta( S, K, T, r, sigma ):
    eps=128.
    cp = BS_price( S, K, T*(1+eps), r, sigma )
    cm = BS_price( S, K, T*(1-eps), r, sigma )
    pass

if __name__ == "__main__":
    interest = 0.02
    
    
    def vol2(K,T):
        return vol(T,K)
    
    PlotFunc3D( vol2, xmin=.8, xmax=1.2, ymin=0.001, ymax=5., labels=["K/S", "time to maturity", "vol"] )
    
    
    
    from functools import partial
    def C_surface( K, T, S, r ):
        v = vol(T, K/S)
        return BS_price( S=S, K=K, T=T, r=r, sigma=v )
    
    price = partial( C_surface, S=10., r=interest )
    PlotFunc3D( price, xmin=8., xmax=12., ymin=0.001, ymax=5., labels=["K", "time to maturity", "call price"] )
    
    
    
    def local_vol( T, K, call_price, S, r, eps=1/128. ):
        call = partial( call_price, S=S, r=r )
        
        parT = ( call(T=T*(1+eps), K=K) - call(T=T*(1-eps), K=K) )/(2*T*eps)
        C = call( T=T, K=K )
        Cp = call( T=T, K=K*(1+eps) )
        Cm = call( T=T, K=K*(1-eps) )
        parK = (Cp-Cm)/(2*K*eps)
        parK2 = (Cp+Cm-2*C)/((K*eps)**2)
        return np.sqrt( (parT + r*K*parK)/( 0.5*K*K*parK2 ) )
    
    def local_vol2(K,T,call_price,S,r):
        return local_vol(T,K,call_price,S,r)
    
    PlotFunc3D( partial( local_vol2, call_price=C_surface, S=10., r=interest ), 
               xmin=8., xmax=12., ymin=0.02, ymax=5, labels=["S", "t", "local vol"] )
    
    
    import local_vol as lv
    from importlib import reload
    reload( lv )
    from local_vol import PathGen
    S0 = 10.
    K = 11.
    T = 1.
    N_T = 1000
    N =50000
    
    
    def forward_vol( T1, S1, Ks, T ):
        localvol = PathGen( N, N_T)
        def forward_local_vol( t, S ):
            return local_vol( T=t+T1, K=S, call_price=C_surface, S=S1, r=interest  )
            
        localvol.params_input( S0=S1, T=T-T1, r=interest, q=0.00, 
                              sigma=forward_local_vol )
        localvol.evolve()
        ST = localvol.S_final.copy()
        res = []
        for k in Ks:
            payoff = np.where( ST>k, ST-k, 0 )
            c = np.mean(payoff)*np.exp(-interest*T)
            #print( c, C_surface( K=k, T=T, S=S1, r=interest ) )
            res.append(c)
        return res
    
    
    import pandas as pd

    for S1 in [8,9,10,11,12]:    
        df = pd.DataFrame()
        Ks = np.linspace(0.8, 1.2, 101)*S1
        df['Ks'] = Ks
        for i in range(7):
            T1 = 0.1*i
            Tf = T1+1.
            calls = forward_vol( T1, S1, Ks, T=Tf-T1 )
            implied_vols = [BS_call_implied_vol(S=S1, K=k, T=Tf-T1, r=interest, C=c) for k, c in zip(Ks, calls) ]
            df['T1='+str(T1)] = implied_vols.copy()
            print(T1, ' finishes')
    #        dir_imp_vols = [vol(Tf, k/S1 ) for k in Ks]
    #        print( implied_vols )
    #        print( np.array(dir_imp_vols) - np.array(implied_vols) )
        df.to_csv('forward vol_'+'S1='+str(S1)+'.csv',  index=False)
            
    
    
    
    
#    constvol = PathGen( N, N_T)
#    constvol.params_input( S0=S0, T=T, r=interest, q=0.00, 
#                          sigma=lambda x,y: ATM_VOL )
#    constvol.evolve()
#    ST = constvol.S_final.copy()    
#    payoff = np.where( ST>K, ST-K, 0 )
#    print( np.mean(payoff)*np.exp(-interest*T), C_surface( K=K, T=T, S=S0, r=interest ) )
    
    
    
    
    
    
    
    
    
    
    
    
    
