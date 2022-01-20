#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 14:45:42 2022

@author: dustinherrmann
"""

# import section 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import numpy as np
import os
from useful_functions import running_percentile
import dill
from scipy.io import savemat


# define session parameters 
animal = 1128
session = 6
n_planes = 4

# navigate to directory and read .p files I need 
load_direc = f'I:\somaDendriteANN_registr\M{animal}\S{session}\suite2p\combined/'
save_direc = f'H:\\Results\Mouse{animal}\S{session}\AnalysisOutputs/'

# load_direc = f'/Users/dustinherrmann/Downloads/combined'
# save_direc = f'/Users/dustinherrmann/Downloads/outputHere'

if not os.path.exists(save_direc):
    os.makedirs(save_direc)

# load stat.py and ops.py files 
stat = np.load(load_direc+'stat.npy',allow_pickle=1)
ops = np.load(load_direc+'ops.npy',allow_pickle=1).item()
iscell = np.load(load_direc+'iscell.npy',allow_pickle=1)
F = np.load(load_direc+'F.npy',allow_pickle=1)
Fneu = np.load(load_direc+'Fneu.npy',allow_pickle=1)

# find cells and planes
num_rois = np.size(np.nonzero(iscell[:,0]))
roi_ids = np.nonzero(iscell[:,0])
stat_iscell = stat[np.nonzero(iscell[:,0])]

F_rois = np.zeros((1,len(F[0])))
F_rois[:] = np.NaN
Fneu_rois = np.zeros((1,len(F[0])))
Fneu_rois[:] = np.NaN
planes_rois = [] 

# iterate over ROIs and extract plane, F and Fneu. Also plot 
im = np.zeros((ops['Ly'], ops['Lx']))
   
for nn in range(0,num_rois):
    #print(nn)
    planes_rois = np.hstack((planes_rois,int(stat_iscell[nn]['iplane'])))
    F_rois = np.concatenate((F_rois,[F[roi_ids[0][nn]]]),axis=0) # option 2 
    Fneu_rois = np.concatenate((Fneu_rois,[Fneu[roi_ids[0][nn]]]),axis=0) # option 2 

    ypix = stat[roi_ids[0][nn]]['ypix']#[~stat[are_dendrites[n]]['overlap']]
    xpix = stat[roi_ids[0][nn]]['xpix']#[~stat[are_dendrites[n]]['overlap']]

    im[ypix,xpix] = int(stat_iscell[nn]['iplane'])+1+nn

ROI_figure = plt.figure()
plt.imshow(im)
plt.set_cmap('Greys_r')
plt.plot(np.r_[0,1024], np.r_[512,512], color='black', linewidth=2)
plt.plot(np.r_[512,512], np.r_[0,1024],  color='black', linewidth=2)
plt.xlim(1, 1024), plt.ylim(1024, 1)
plt.show()


# delete the first row that was used to initialize 
F_rois = np.delete(F_rois,obj=0, axis=0)
Fneu_rois = np.delete(Fneu_rois,obj=0, axis=0)

# compute dFF using rolling percentile 
# dataframe_perRoi = pd.DataFrame(F_rois)
# F0 = dataframe_perRoi.T.rolling(window=1000).quantile(0.1) 
# F0 = F0.T.to_numpy();
# last_idx = max(np.nonzero(np.isnan(F0[0,:]))[0])
# F0[:,0:last_idx] = np.tile(F0[:,last_idx+1], (last_idx, 1)).T
F0 = running_percentile(F_rois,1000,10)
dFF = (F_rois-F0)/F0;


# plt.plot(F0[0,:])
# plt.plot(F_rois[0,:])
# plt.show()


# plot dF/F traces 
x = np.ndarray.astype(np.unique(planes_rois),'int')
xx = np.ndarray.astype(np.linspace(0,255,len(x)),'int')
this_cmap = cm.plasma(xx)
use_cmap = this_cmap[np.ndarray.astype(np.r_[planes_rois],'int'),:]
use_cmap = np.delete(use_cmap,obj=3, axis=1)

F_figure = plt.figure()
for nn in range(0,num_rois):
    plt.plot(dFF[nn,:],color=use_cmap[int(planes_rois[nn]),:],label=f'dFF_ROI {nn} plane {int(planes_rois[nn])}')
    plt.legend()
    
plt.show()


F_figure



# to do: 
    # save F, Fneu, planes into .mat file 
    # save figures 
    # write corresponding matlab script to read files and plug them into the analysis

# save important variables in .mat files 
savemat(save_direc+f'py2p_M{animal}_S{session}.mat', {"dFF": dFF,"F0": F0, 
                                                      "F":F_rois, "FNeu":Fneu_rois,
                                                      "planes":planes_rois, 
                                                      "image":im, "ops":ops,
                                                      "stat":stat, "iscell":iscell})

#  save entire session 
dill.dump_session(save_direc+f'py2p_workspace_M{animal}_S{session}.pkl') # save entire session
# dill.load_session('test.pkl') # this is how you load the entire session later on 

# plt.imshow(ops['refImg'])
# plt.set_cmap('Greys_r')

# plt.imshow(ops['meanImg'])
# plt.set_cmap('Greys_r')
# plt.clim(0,100)


# TO DO: IS THERE A WEIGHTING FACTOR FOR THE NP CORRECTION?



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

                

                

                

                

                

                
