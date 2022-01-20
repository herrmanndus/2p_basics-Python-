# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:14:51 2022

@author: Dustin
"""

def running_percentile(traces,wndw,prctl):
    
    import numpy as np
    
    num_trcs = np.shape(traces)[0]
    num_frames = np.shape(traces)[1]
    
    running_prctl_out = np.empty(np.shape(traces))
    running_prctl_out[:] = np.NaN
         
    for t in range(0,num_trcs):
        thisF = traces[0,:]
        i = 0
        while i < len(thisF):
            wndw_ind = np.arange((i-wndw/2),(i+wndw/2))
            wndw_ind = wndw_ind[np.nonzero(wndw_ind >= 0) and np.nonzero(wndw_ind < num_frames)]
            wndw_vals = thisF[np.ndarray.astype(wndw_ind,'int')];
            prct_wndw = np.percentile(wndw_vals, prctl)
            running_prctl_out[t,i] = prct_wndw
            i += 1
            
    return running_prctl_out
