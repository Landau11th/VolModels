# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:23:50 2019

@author: e0008730
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

from vol_surface import vol

def plot_and_save( S1 ):
    folder = ""#"10000samples"
    prefix = "forward vol_S1="
    filetype = ".csv"
    df = pd.read_csv( os.path.join( folder, prefix+str(S1)+filetype ) )
    
    x_name = "Ks"
    Ks = df[x_name]
    Tf = 1.
    print(S1)
    impl_vol_surface = vol(Tf, Ks/S1 )
    fig, ax = plt.subplots()
    fig.suptitle("Forward skew at T1 with S1="+str(S1), fontsize=16)
    ax.plot( Ks/S1, impl_vol_surface, label="implied vol\n at t=0" )
    
    new_col = [ c[:6] for c in df ]
    df.columns = new_col

    for col in df:
        if col.startswith("T1") and col[-1]!='6':
            ax.plot( Ks/S1, df[col] )
    lgd = ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    ax.set_xlabel("K/S")
    ax.set_ylabel("forward vol with maturity 1.0")
    fig.savefig("forward skew "+str(int(S1))+".jpg", dpi=400, bbox_inches='tight')
    pass    
    
if __name__ == "__main__":
    toplot = [8.,10.,12.]
    [plot_and_save(s) for s in toplot]
 
    
    
    
    
    
    
    
    
    
    
    
    