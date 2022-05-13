#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:36:37 2021

@author: ken38
"""



#  PROCESSING NF GRAINS WITH MISORIENTATION
#==============================================================================
import numpy as np

import matplotlib.pyplot as plt

import multiprocessing as mp

import os

#from hexrd.grainmap import tomoutil
import numpy as np
import scipy as sp

import scipy.ndimage as img
try:
    import imageio as imgio
except(ImportError):
    from skimage import io as imgio
import skimage.transform as xformimg

import tomoFunctions2 as tf
from skimage.transform import iradon, radon, rescale
import tomopy

#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
#Locations of tomography dark field images
#tdf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/3/nf/'
tdf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/df_1mm_0508-1/3/nf/'
tdf_img_start=6060#8223 #for this rate, this is the 6th file in the folder
tdf_num_imgs=1

#Locations of tomography bright field images
#tbf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/2/nf/'
tbf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/bf_1mm_0508-1/1/nf/' #bf_1mm_0508-1
tbf_img_start=5928#8203 #for this rate, this is the 6th file in the folder
tbf_num_imgs=20

#Locations of tomography images
#tomo_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/1/nf/'
tomo_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/tomo_1mm_0508-1/3/nf/'
tomo_img_start=4484#7843#for this rate, this is the 6th file in the folder
tomo_num_imgs=300#1440
#%%============================================================================
#% DETECTOR PARAMETERS
#==============================================================================
nrows = 3008
ncols = 4112
pixel_size = 0.00345/5

#SCAN PARAMETERS
#ome_range_deg = [(0.,359.0)]
start_tomo_ang = 0.0
end_tomo_ang= 180.0
tomo_num_imgs = 720

#SAMPLE DIMENSIONS
cross_sectional_dim = 1.8#in mm
#==============================================================================
# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
#==============================================================================
img_y_bounds = np.array([0,ncols])
img_x_bounds = np.array([595,2450])

#%%============================================================================
#%GENERATE DARK, BRIGHT FIELD AND RADIOGRAPHS
#==============================================================================
tdf = tf.genDark(tdf_data_folder,tdf_img_start,0, tdf_num_imgs)
where = np.where(tdf > 21) #*1.4#
for i in range(0,where[0].size):
    tdf[where[0][i],where[1][i]] = 18
plt.imshow(tdf)

#%%
#tdf = tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
tbf = tf.genBright(tbf_data_folder,tdf,tbf_img_start,0, tbf_num_imgs)#*1.4
#tbf = np.zeros([tbf.shape[0],tbf.shape[1]])
#tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
intCorr = np.zeros([tomo_num_imgs])+1.
theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
rad_stack_0 = tf.genTomo(tomo_data_folder, tdf, tbf,img_x_bounds,img_y_bounds, intCorr, tomo_img_start, 0, tomo_num_imgs, theta)

#%%
#CHOOSE BOUNDS FROM RADIOGRAPH FOR RECONSTRUCTION (ROI)
rad_stack = rad_stack_0[0,:,:,:]
rad_stack = np.swapaxes(rad_stack,0,1)

#%%
zoom_perc = 0.55
red_stack_z = img.zoom(rad_stack[:,:,:],(1,zoom_perc,zoom_perc))

np.save('red_stack_55p_bot.npy',red_stack_z)

#%%#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
#Locations of tomography dark field images
#tdf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/3/nf/'
tdf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/df_1mm_0508-1/3/nf/'
tdf_img_start=6060#8223 #for this rate, this is the 6th file in the folder
tdf_num_imgs=1

#Locations of tomography bright field images
#tbf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/2/nf/'
tbf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/bf_1mm_0508-1/1/nf/' #bf_1mm_0508-1
tbf_img_start=5928#8203 #for this rate, this is the 6th file in the folder
tbf_num_imgs=20

#Locations of tomography images
#tomo_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/1/nf/'
tomo_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/tomo_1mm_0508-1/1/nf/'
tomo_img_start=1767#7843#for this rate, this is the 6th file in the folder
tomo_num_imgs=720#1440

#==============================================================================
# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
#==============================================================================
img_y_bounds = np.array([0,ncols])
img_x_bounds = np.array([595,2450])

#%%============================================================================
#%GENERATE DARK, BRIGHT FIELD AND RADIOGRAPHS
#==============================================================================
tdf = tf.genDark(tdf_data_folder,tdf_img_start,0, tdf_num_imgs)
where = np.where(tdf > 21) #*1.4#
for i in range(0,where[0].size):
    tdf[where[0][i],where[1][i]] = 18
plt.imshow(tdf)

#%%
#tdf = tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
tbf = tf.genBright(tbf_data_folder,tdf,tbf_img_start,0, tbf_num_imgs)#*1.4
#tbf = np.zeros([tbf.shape[0],tbf.shape[1]])
#tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
intCorr = np.zeros([tomo_num_imgs])+1.
theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
rad_stack_0 = tf.genTomo(tomo_data_folder, tdf, tbf,img_x_bounds,img_y_bounds, intCorr, tomo_img_start, 0, tomo_num_imgs, theta)

#%%
#CHOOSE BOUNDS FROM RADIOGRAPH FOR RECONSTRUCTION (ROI)
rad_stack = rad_stack_0[0,:,:,:]
rad_stack = np.swapaxes(rad_stack,0,1)

#%%
zoom_perc = 0.55
red_stack_z = img.zoom(rad_stack[:,:,:],(1,zoom_perc,zoom_perc))

np.save('red_stack_55p_top.npy',red_stack_z)

#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
#Locations of tomography dark field images
#tdf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/3/nf/'
tdf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/df_1mm_0508-1/3/nf/'
tdf_img_start=6060#8223 #for this rate, this is the 6th file in the folder
tdf_num_imgs=1

#Locations of tomography bright field images
#tbf_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/2/nf/'
tbf_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/bf_1mm_0508-1/1/nf/' #bf_1mm_0508-1
tbf_img_start=5928#8203 #for this rate, this is the 6th file in the folder
tbf_num_imgs=20

#Locations of tomography images
#tomo_data_folder='/nfs/chess/id1a3/2019-2/kacher-951-1/sample3-5003/1/nf/'
tomo_data_folder='/nfs/chess/raw/2021-2/id1a3/ramshaw-3168-A/tomo_1mm_0508-1/2/nf/'
tomo_img_start=3040#7843#for this rate, this is the 6th file in the folder
tomo_num_imgs=720#1440
#==============================================================================
# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
#==============================================================================
img_y_bounds = np.array([0,ncols])
img_x_bounds = np.array([595,2450])

#%%============================================================================
#%GENERATE DARK, BRIGHT FIELD AND RADIOGRAPHS
#==============================================================================
tdf = tf.genDark(tdf_data_folder,tdf_img_start,0, tdf_num_imgs)
where = np.where(tdf > 21) #*1.4#
for i in range(0,where[0].size):
    tdf[where[0][i],where[1][i]] = 18
plt.imshow(tdf)

#%%
#tdf = tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
tbf = tf.genBright(tbf_data_folder,tdf,tbf_img_start,0, tbf_num_imgs)#*1.4
#tbf = np.zeros([tbf.shape[0],tbf.shape[1]])
#tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]
intCorr = np.zeros([tomo_num_imgs])+1.
theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
rad_stack_0 = tf.genTomo(tomo_data_folder, tdf, tbf,img_x_bounds,img_y_bounds, intCorr, tomo_img_start, 0, tomo_num_imgs, theta)

#%%
#CHOOSE BOUNDS FROM RADIOGRAPH FOR RECONSTRUCTION (ROI)
rad_stack = rad_stack_0[0,:,:,:]
rad_stack = np.swapaxes(rad_stack,0,1)

#%%
zoom_perc = 0.55
red_stack_z = img.zoom(rad_stack[:,:,:],(1,zoom_perc,zoom_perc))

np.save('red_stack_55p_mid.npy',red_stack_z)
