# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:23:50 2019

@author: e0008730
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vol_surface import vol

if __name__ == "__main__":
    S1 = 8.
    df = pd.read_csv( "forward volS1="+str(S1)+".csv" )
    
    
    x_name = "Ks"
    Ks = df[x_name]
    Tf = 1.
    print(S1)
    impl_vol_surface = vol(Tf, Ks/S1 )
    plt.plot( Ks, impl_vol_surface )
    
    new_col = [ c[:6] for c in df ]
    df.columns = new_col
    
    for col in df:
        if col.startswith("T1"):
            plt.plot( x_name, col, data=df )
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        
        
    
    
    
    
    
    
    
    
    
    
    
    