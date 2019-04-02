# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:11:26 2019

@author: e0008730
"""
import numpy as np
import types

class PathGen():
    def __init__( self, sample_size:int, N_t:int ):
        self._N = sample_size
        self._N_t = N_t
        self._if_params_input = False
    
    def params_input( self, T:float, S0:float, r:float, 
                     q:float, sigma:types.FunctionType ):
        self.T = T
        self.dt = self.T/self._N_t
        self.dt_sqrt = np.sqrt(self.dt)
        self.S0 = S0
        self.r = r
        self.q = q
        self.sigma = sigma
        
        self._if_params_input = True
        self.S_init = np.full( (self._N), self.S0 )
        self.S_final = self.S_init.copy()
    
    def evolve( self ):
        if not self._if_params_input:
            raise ValueError( "Parameters not complete: params_input is not called" )
        
        for i in range( self._N_t ):
            self._one_step(i)
        
    def _one_step( self, i:int ):
        #print(self.S0)
        dt_term = (self.r-self.q)*self.S_init*self.dt
        dW = np.random.normal(scale=self.dt_sqrt, size=self._N)
        dW_coeff = self.sigma( (i+0.01)*self.dt, self.S_init )*self.S_init
        S_hat = self.S0 + (dt_term + dW_coeff*self.dt_sqrt)
        self.S_final = self.S_init
        self.S_final += dt_term + dW_coeff*dW + 0.5*( S_hat-self.S0 )*(dW*dW/self.dt_sqrt - self.dt_sqrt)
        self.S0 = self.S_final
    
    
if __name__ == "__main__":
    localvol = PathGen( 3, 10 )
    localvol.params_input( S0=10., T=1., r=0.03, q=0.01, sigma=lambda x, y: 0.5*x+0.01*y )
    localvol.evolve()
    print( localvol.S_final )
    
    
    
    