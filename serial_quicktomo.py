#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 07:43:07 2020

@author: ken38
"""

#%% Necessary Dependencies
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
#%%
def do_reconstruction_matrix (layer, stack, recon_matrix, centerlist):
    print layer
    center = center_list[layer]*effective_pixel
    sinogram=np.squeeze(stack[:,layer,:])
    rotation_axis_pos=-int(np.round(center/effective_pixel))
    theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
    max_rad=int(cross_sectional_dim/effective_pixel/2.*1.1) #10% slack to avoid edge effects

    if rotation_axis_pos>=0:
        sinogram_cut=sinogram[:,2*rotation_axis_pos:]
    else:
        sinogram_cut=sinogram[:,:(2*rotation_axis_pos)]
    dist_from_edge=np.floor(sinogram_cut.shape[1]/2.).astype(int)-max_rad                          
    
    sinogram_cut=sinogram_cut[:,dist_from_edge:-dist_from_edge]
    
    print('Inverting Sinogram....')
    reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
    gauss1 = img.gaussian_filter(reconstruction_fbp,0.5)
#gauss1 = filters.gaussian(reconstruction_fbp,3.0)
    recons = np.zeros([1,gauss1.shape[0],gauss1.shape[1]])
    recons[0,:] = gauss1
    recon_clean=tomopy.misc.corr.remove_ring(recons,rwidth=17)

    recon_matrix[layer,:,:]=recon_clean[0,xlow:xhigh,ylow:yhigh]

    return recon_matrix


def do_reconstruction_layer (layer, stack, center):
    print layer
    center = center*effective_pixel
    sinogram=np.squeeze(stack[:,layer,:])
    rotation_axis_pos=-int(np.round(center/effective_pixel))
    theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
    max_rad=int(cross_sectional_dim/effective_pixel/2.*1.1) #10% slack to avoid edge effects

    if rotation_axis_pos>=0:
        sinogram_cut=sinogram[:,2*rotation_axis_pos:]
    else:
        sinogram_cut=sinogram[:,:(2*rotation_axis_pos)]
    dist_from_edge=np.floor(sinogram_cut.shape[1]/2.).astype(int)-max_rad                          
    
    sinogram_cut=sinogram_cut[:,dist_from_edge:-dist_from_edge]
    
    print('Inverting Sinogram....')
    reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
    gauss1 = img.gaussian_filter(reconstruction_fbp,0.5)
#gauss1 = filters.gaussian(reconstruction_fbp,3.0)
    recons = np.zeros([1,gauss1.shape[0],gauss1.shape[1]])
    recons[0,:] = gauss1
    recon_clean=tomopy.misc.corr.remove_ring(recons,rwidth=17)

    return recon_clean

def center_finder_matrix (layer, stack, recon_matrix, center0):
    print center0
    center = center0*effective_pixel
    sinogram=np.squeeze(stack[:,layer,:])
    rotation_axis_pos=-int(np.round(center/effective_pixel))
    theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
    max_rad=int(cross_sectional_dim/effective_pixel/2.*1.1) #10% slack to avoid edge effects

    if rotation_axis_pos>=0:
        sinogram_cut=sinogram[:,2*rotation_axis_pos:]
    else:
        sinogram_cut=sinogram[:,:(2*rotation_axis_pos)]
    dist_from_edge=np.floor(sinogram_cut.shape[1]/2.).astype(int)-max_rad                          
    
    sinogram_cut=sinogram_cut[:,dist_from_edge:-dist_from_edge]
    
    print('Inverting Sinogram....')
    reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
    gauss1 = img.gaussian_filter(reconstruction_fbp,0.5)
#gauss1 = filters.gaussian(reconstruction_fbp,3.0)
    recons = np.zeros([1,gauss1.shape[0],gauss1.shape[1]])
    recons[0,:] = gauss1
    recon_clean=tomopy.misc.corr.remove_ring(recons,rwidth=17)

    recon_matrix[center0-test_centers[0],:,:]=recon_clean[0,:,:]

    return recon_matrix

#%%##################JUMP TO LOADED DATA IF YOU HAVE IT########################
    #else do the following blocks
###############################################################################

#%%============================================================================
#% DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
#==============================================================================
    
cycle = '2021-2'
beamline = 'id3a'
btr = 'hassani-1000-5'
sampleID = 'He-2/Tomo'

#Locations of tomography dark field images
darkfield_scann = 2
tdf_data_folder='/nfs/chess/raw/'+cycle+'/'+beamline+'/'+btr+'/'+sampleID+'/'+str(darkfield_scann)+'/nf/'
tdf_img_start = 151267
tdf_num_imgs = 20

#Locations of tomography bright field images
brightfield_scann = 3
tbf_data_folder = '/nfs/chess/raw/'+cycle+'/'+beamline+'/'+btr+'/'+sampleID+'/'+str(brightfield_scann)+'/nf/'
tbf_img_start = 151293
tbf_num_imgs = 20

#Locations of tomography images
tomo_scann = 4
tomo_data_folder ='/nfs/chess/raw/'+cycle+'/'+beamline+'/'+btr+'/'+sampleID+'/'+str(tomo_scann)+'/nf/'
tomo_img_start = 151319
tomo_num_imgs = 1440

#%%============================================================================
#% DETECTOR PARAMETERS
#==============================================================================
nrows = 2048
ncols = 2048
pixel_size = 1.48

#SCAN PARAMETERS
start_tomo_ang = 0.0
end_tomo_ang= 180.0
tomo_num_imgs = 1440

#SAMPLE DIMENSIONS
cross_sectional_dim = 2.5#in mm
#%%==============================================================================
# SET BOUNDS FOR IMAGE STACK
#==============================================================================
img_y_bounds = np.array([0,ncols])
img_x_bounds = np.array([0,nrows]) # I start with entire image ([0,nrows]) before setting these bounds for X-ray field of view
#^reducing the x-bounds saves memory in the computer so rerun when bounds are found. 
#%%============================================================================
#%GENERATE DARK
#==============================================================================
tdf = tf.genDark(tdf_data_folder,tdf_img_start,0, tdf_num_imgs)

#for my dataset I had an issue with my df correction so needed to add below - this should not be necessary for most datasets. 
where = np.where(tdf > 21) #*1.4#
for i in range(0,where[0].size):
    tdf[where[0][i],where[1][i]] = 18
plt.imshow(tdf)

#%%#%%============================================================================
#%GENERATE BRIGHT FIELD AND RADIOGRAPHS
#==============================================================================
#tdf = tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]] #moved to block above because of correction
tbf = tf.genBright(tbf_data_folder,tdf,tbf_img_start,0, tbf_num_imgs)
intCorr = np.zeros([tomo_num_imgs])+1.
theta = np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False)
rad_stack_0 = tf.genTomo(tomo_data_folder, tdf, tbf,img_x_bounds,img_y_bounds, intCorr, tomo_img_start, 0, tomo_num_imgs, theta)

#%%PLOT RAD_STACK TO DETERMINE ROI BOUNDS - iterate back to "SET BOUNDS FOR IMAGE STACK" and rerun to keep image stack small

img_layer = 570
plt.figure('rad_stack_0_fullres')
plt.imshow(rad_stack_0[0,:,img_layer,:],vmin=-1.0,vmax=1.2)

#%% CONVERT RADSTACK_0 INTO CORRECT SHAPE AND AXIS FOR QUICK TOMO

rad_stack = rad_stack_0[0,:,:,:]
rad_stack = np.swapaxes(rad_stack,0,1)

#%% VISUALIZE RAD_STACK RADIOGRAPHS TO CONFIRM 

img_layer = 10
plt.figure('rad_stack_fullres_check')
plt.imshow(rad_stack[img_layer,:,:],vmin=0.0,vmax=1.2)
#%%    
#screen out nans:
nan_val = np.argwhere(np.isnan(rad_stack))
for iii in range(0,nan_val.shape[0]):    
    rad_stack[nan_val[iii][0],nan_val[iii][1],nan_val[iii][2]] = 0.
#%% ZOOM FUNCTION "DOWNSAMPLES" - INTERPOLATES THE ARRAY TO % OF FULL RESOLUTION
#IF YOU WANT TO DOWNSAMPLE - WE PROBABLY DON'T WANT THIS IF USING THE RETIGA
#must leave first column at 1 > this is the angle - you do not want to change the resolution on this
#second two columns currently making image 69% of full resolution image - my pixel size was 0.0069 mm and it will now effectively be 0.001
#reduction_factor = 0.69 #percentage of full resolution image
#red_stack_z = img.zoom(rad_stack[:,:,:],(1,reduction_factor,reduction_factor)) #this can take awhile. go get coffee/lunch

#IF YOU DON'T WANT TO DOWNSAMPLE:
red_stack_z = rad_stack
#%% check loser resolution image 

plt.figure('rad_stack_lowerres_check')
plt.imshow(red_stack_z[img_layer,:,:] )

#%% SAVE DATA IF YOU ARE HAPPY SO YOU DON'T NEED TO DO ABOVE
np.save('red_stack_z_mid.npy',red_stack_z)

#%%############################################################################ 
##########################LOAD SAVED DATA IF YOU HAVE IT!!#####################
###############################################################################
red_stack_z = np.load('red_stack_z_mid.npy')

#%%RECONSTRUCT ONE LAYER FOR FINDING CENTER AND IMAGE PROCESSING PARAMETERS
reduction_factor = 1
effective_pixel=pixel_size/reduction_factor
stack = red_stack_z
center_pixel_dist = 26# FAST JUST CHANGE AND RUN WITH DIFFERENT VALUES
layer_row = 70 #row on image to reconstruct

recon_layer = do_reconstruction_layer(layer_row, stack, center_pixel_dist)
plt.figure('reconstruction_layer')
plt.imshow(recon_layer[0,:,:])

#=======ITERATE ABOVE TO FIND ROTATION CENTER AT TOP AND BOTTOM OF RAD STACK===
#%%============================================================================
stack = red_stack_z #reduced image stack

xlow = 200 #column bound from recon_layer
xhigh = 1400 #column bound from recon_layer
ylow = 1000 #row bound from recon_layer
yhigh = 1500 #row bound from recon_layer

center_bnds = [36,29] #top and bottom centers
center_list= np.linspace(center_bnds[0],center_bnds[1],stack.shape[1]) #center list for each point in space
recon_matrix = np.zeros([stack.shape[1],xhigh-xlow,yhigh-ylow]) #empty full reconstruction matrix
#%% serial processing of datasets
for i in range(0,stack.shape[1]):
    layer = i
    recon_matrix = do_reconstruction_matrix(layer, stack, recon_matrix, center_list)

np.save('recon_matrix_080521.npy',recon_matrix)

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#%% FOR LOOP FOR DETERMINING CENTER
#==============================================================================
test_centers = np.arange(25,26,1) #LIST OF TEST PIXEL VALUES FOR CENTER
layer = 300 #select layer to find zeros on
cent_find_matrix = np.zeros([len(test_centers),reconstruction_fbp.shape[0],reconstruction_fbp.shape[1]])

#%%
for ii in range(0,len(test_centers)):
    center=test_centers[ii]
    cent_find_matrix = center_finder_matrix(layer, stack ,cent_find_matrix , center)

#%%
center_id = 0
plt.figure('center %d' % test_centers[center_id])
plt.imshow(cent_find_matrix[center_id,:,:])  

#%%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# tomopy with SIRT
stack = stack1#red_stack_z
sinograms = np.zeros([1,stack.shape[0],stack.shape[1],stack.shape[2]])
sinograms[0,:,:] = stack
sinograms = np.swapaxes(sinograms,1,2)
#rad_stack_0[0,:,:,:]
#rad_stack = np.swapaxes(rad_stack,0,1)
#%%
#sinograms = rad_stack_0
plt.figure()
imageBounds = [0,stack.shape[2]]#img_y_bounds
layers = [56,853]#[70,1070]#img_x_bounds
theta = np.deg2rad(np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False))
topVals = tf.launchValHelper(sinograms, imageBounds, layers[1],layers, theta, sigma = .4, ncore = 24)# , vmin=-0.0001, vmax=.0005)
#%%
#%%
sinograms2 = np.zeros([1,stack2.shape[0],stack2.shape[1],stack2.shape[2]])
sinograms2[0,:,:] = stack2
sinograms2 = np.swapaxes(sinograms2,1,2)

#sinograms = rad_stack_0
imageBounds = [0,stack2.shape[2]]#img_y_bounds
layers = [56,853]#[70,1070]#img_x_bounds
theta = np.deg2rad(np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False))
topVals = tf.launchValHelper(sinograms2, imageBounds, layers[0],layers, theta, sigma = .4, ncore = 24)# , vmin=-0.0001, vmax=.0005)
#%%
topCenter = -23#-36
bottomCenter = -23#-29
layers = np.array([853,854])
centers = tf.calcCenters(layers,topCenter, bottomCenter) 
secondary_iterations = 50

print('Reconstructing')
recon_clean1 = tf.reconstruct(sinograms, centers, imageBounds, layers, 
                 theta, ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
#%%
sinograms2 = sinograms
topCenter2 = -30#-36
bottomCenter2 = -30#-29
layers2 = np.array([400,401])
centers2 = tf.calcCenters(layers2,topCenter2, bottomCenter2) 
secondary_iterations = 50

print('Reconstructing')
recon_clean2 = tf.reconstruct(sinograms2, centers2, imageBounds, layers2, 
                 theta, ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)

#%
plt.figure('overlay')
#plt.imshow(recon_clean1[0,0,:,:],vmin=0,vmax=0.003)
plt.imshow(recon_clean2[0,0,:,:],alpha=1.0, cmap='magma',vmin=0, vmax=0.003)


#%%
topCenter = -36#-36
bottomCenter = -29
layers = np.array([70,1070])
secondary_iterations = 100
centers = tf.calcCenters(layers,topCenter, bottomCenter) 
reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
                    theta, sigma = 0.1 , ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
np.save('mid_lay_recon.npy', reconstruction)
#%%
from skimage.restoration import denoise_tv_chambolle

edges1 = denoise_tv_chambolle(reconstruction, weight = 0.001)
#%%
plt.close('all')
plt.figure()
plt.imshow(edges1[0,0,:,:], vmin=0.0004, vmax = 0.002)