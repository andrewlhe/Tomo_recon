#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 07:43:07 2020

@author: ken38
"""

# %% Necessary Dependencies
# ==============================================================================
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
import skimage.filters as filters
import tomopy
from sys import platform

# %%============================================================================
# % DATA LOCATIONS - 3A AND ID1A3 SAVING SCHEMA
# ==============================================================================


# CLASSE directory
if platform == "linux":
    cycle = '2021-2'
    beamline = 'id3a'
    btr = 'hassani-1000-5'
    sampleID = 'He-2/Tomo'
    # Locations of tomography dark field images
    darkfield_scann = 2
    tdf_data_folder = '/nfs/chess/raw/'+cycle+'/'+beamline + \
        '/'+btr+'/'+sampleID+'/'+str(darkfield_scann)+'/nf/'
    # Locations of tomography bright field images
    brightfield_scann = 3
    tbf_data_folder = '/nfs/chess/raw/'+cycle+'/'+beamline + \
        '/'+btr+'/'+sampleID+'/'+str(brightfield_scann)+'/nf/'
    # Locations of tomography images
    tomo_scann = 4
    tomo_data_folder = '/nfs/chess/raw/'+cycle+'/'+beamline + \
        '/'+btr+'/'+sampleID+'/'+str(tomo_scann)+'/nf/'

# local directory
else:
    # Locations of tomography dark field images
    tdf_data_folder = 'Y://CHESS//ID3A_2021-2//raw data//Tomo//He-2-Tomo//2//nf//'
    # Locations of tomography bright field images
    tbf_data_folder = 'Y://CHESS//ID3A_2021-2//raw data//Tomo//He-2-Tomo//3//nf//'
    # Locations of tomography images
    tomo_data_folder = 'Y://CHESS//ID3A_2021-2//raw data//Tomo//He-2-Tomo//4//nf//'

tdf_img_start = 151267
tdf_num_imgs = 20
tbf_img_start = 151293
tbf_num_imgs = 20
tomo_img_start = 151319
tomo_num_imgs = 1440


# %%============================================================================
# % DETECTOR PARAMETERS
# ==============================================================================
nrows = 2048
ncols = 2048
pixel_size = 0.00148

# SCAN PARAMETERS
start_tomo_ang = 0.0
end_tomo_ang = 360.0
tomo_num_imgs = 1440

# SAMPLE DIMENSIONS
cross_sectional_dim = 1.8  # in mm
# ==============================================================================

# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
# ==============================================================================
img_x_bounds = np.array([0, nrows])
#img_y_bounds = np.array([250,1798])
#img_x_bounds = np.array([650,1350])
img_y_bounds = np.array([0, ncols])
#img_x_bounds = np.array([750,1250])

# %%============================================================================
# %GENERATE DARK, BRIGHT FIELD AND RADIOGRAPHS
# ==============================================================================
# tdf will be the mean darkfield:
tdf = tf.genDark(tdf_data_folder, tdf_img_start, 0, tdf_num_imgs)
#tdf = tdf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

# tbf will be the mean bright field:
tbf = tf.genBright(tbf_data_folder, tdf, tbf_img_start, 0, tbf_num_imgs)
#tbf = tbf[img_x_bounds[0]:img_x_bounds[1],img_y_bounds[0]:img_y_bounds[1]]

# read the tomo images into rad_stack_0, does the dark- and bright-field correction
# rad_stack_0 is of size 1 x nrows x tomo_num_imgs x ncols
intCorr = np.zeros([tomo_num_imgs]) + 1.
theta = np.linspace(start_tomo_ang, end_tomo_ang,
                    tomo_num_imgs, endpoint=False)
rad_stack_0 = tf.genTomo(tomo_data_folder, tdf, tbf, img_x_bounds,
                         img_y_bounds, intCorr, tomo_img_start, 0, tomo_num_imgs, theta)

# %%PLOT RAD_STACK TO DETERMINE ROI BOUNDS
img_layer = 10  # which image out of tomo_num_imgs to show
plt.figure()
plt.imshow(rad_stack_0[0, :, img_layer, :], vmin=0, vmax=1.8)

# %%
# CHOOSE BOUNDS FROM RADIOGRAPH FOR RECONSTRUCTION (ROI)
roi_row_start = 716
roi_row_end = 1398
roi_col_start = 197
roi_col_end = 1882
rad_stack = rad_stack_0[0, roi_row_start:roi_row_end,
                        :, roi_col_start:roi_col_end]
rad_stack = np.swapaxes(rad_stack, 0, 1)

# %%
plt.figure()
plt.imshow(rad_stack[img_layer], vmin=-0.2, vmax=0.5)

# %%
# screen out nans:
nan_val = np.argwhere(np.isnan(rad_stack))
for iii in range(0, nan_val.shape[0]):
    rad_stack[nan_val[iii][0], nan_val[iii][1], nan_val[iii][2]] = 0.

# %%RECONSTRUCT ONE LAYER FOR FINDING CENTER AND IMAGE PROCESSING PARAMETERS
reduction_factor = 1.0
effective_pixel = pixel_size/reduction_factor
stack = rad_stack

# adjust these manually and tinker till results look reasonable
pixel_dist = 22.5  # distance in pixels from the object center to the center of rotation
# center of the sinogram is at (667,667)
layer_row = 500  # which row within the ROI to show a slice through
# SE ROW TO FIND CENTER
cross_sectional_dim = 1.8

center = pixel_dist*effective_pixel
# def tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=1024,start_tomo_ang=0., end_tomo_ang=360.,tomo_num_imgs=360, center=0.,pixel_size=0.00148):
sinogram = np.squeeze(stack[:, layer_row, :])
rotation_axis_pos = -int(np.round(center/effective_pixel))
# rotation_axis_pos=13
theta = np.linspace(start_tomo_ang, end_tomo_ang,
                    tomo_num_imgs, endpoint=False)
max_rad = int(cross_sectional_dim/effective_pixel/2. *
              1.1)  # 10% slack to avoid edge effects

if rotation_axis_pos >= 0:
    sinogram_cut = sinogram[:, 2*rotation_axis_pos:]
else:
    sinogram_cut = sinogram[:, :(2*rotation_axis_pos)]
dist_from_edge = np.round(sinogram_cut.shape[1]/2.).astype(int)-max_rad

sinogram_cut = sinogram_cut[:, dist_from_edge:-dist_from_edge]

print('Inverting Sinogram....')
reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
gauss1 = filters.gaussian(reconstruction_fbp, 0.)
#gauss1 = filters.gaussian(reconstruction_fbp,3.0)
recons = np.zeros([1, gauss1.shape[0], gauss1.shape[1]])
recons[0, :] = gauss1
recon_clean = tomopy.misc.corr.remove_ring(recons, rwidth=17)
#gauss1 = filters.sobel(gauss1)
plt.figure('reconstruction22P5')
plt.imshow(recon_clean[0, :, :], vmin=-.0007,
           vmax=0.0014, cmap='viridis')  # %%

# %% PLOT FIGURE
plt.figure()
plt.imshow(recon_clean[0, :, :], vmin=-.0007, vmax=0.0014, cmap='viridis')

# %% PLOT SINOGRAM AVERAGE FOR HELP WITH ROT CENTER
sinavg = np.average(sinogram, axis=0)
plt.figure()
plt.scatter(np.linspace(
    1, img_x_bounds[1]-img_x_bounds[0], img_y_bounds[1]-img_y_bounds[0]), sinavg)
    
# %% MAKE IMAGE BINARY WITH SoME DEFINED THRESHOLD
bi_img = gauss1 > 0.0006
# binary_recon=morphology.binary_erosion(bi_img,iterations=2)
# binary_recon=img.morphology.binary_dilation(bi_img,iterations=1)
# binary_recon=img.morphology.binary_erosion(binary_recon,iterations=0)
plt.imshow(bi_img)

# %%
stack = stack  # _reduced #image_stack_reduced
pixel_dist = 1
tomo_3d = np.zeros(
    [stack.shape[1], reconstruction_fbp.shape[0], reconstruction_fbp.shape[1]])
    
# %%
for layer in range(0, stack.shape[1]):
    print(layer)
    center = pixel_dist*effective_pixel

# def tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=1024,start_tomo_ang=0., end_tomo_ang=360.,tomo_num_imgs=360, center=0.,pixel_size=0.00148):
    sinogram = np.squeeze(stack[:, layer, :])
    rotation_axis_pos = -int(np.round(center/effective_pixel))
    # rotation_axis_pos=13
    theta = np.linspace(start_tomo_ang, end_tomo_ang,
                        tomo_num_imgs, endpoint=False)

    max_rad = int(cross_sectional_dim/effective_pixel/2. *
                  1.0)  # 10% slack to avoid edge effects

    if rotation_axis_pos >= 0:
        sinogram_cut = sinogram[:, 2*rotation_axis_pos:]
    else:
        sinogram_cut = sinogram[:, :(2*rotation_axis_pos)]
    dist_from_edge = np.round(sinogram_cut.shape[1]/2.).astype(int)-max_rad

    sinogram_cut = sinogram_cut[:, dist_from_edge:-dist_from_edge]

    print('Inverting Sinogram....')
    reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
    gauss1 = filters.gaussian(reconstruction_fbp, 2.0)
    # gauss1 = filters.gaussian(reconstruction_fbp,3.0)
    recons = np.zeros([1, gauss1.shape[0], gauss1.shape[1]])
    recons[0, :] = gauss1
    recon_clean = tomopy.misc.corr.remove_ring(recons, rwidth=17)

    tomo_3d[layer, :, :] = recon_clean

np.save('quicktomo_hassani.npy', tomo_3d)
#gauss = filters.gaussian(reconstruction_fbp, sigma=0.9)
#gauss = filters.laplace(reconstruction_fbp)

# %%
tomo_reduced = tomo_3d.astype('float16')

np.save('tomo_3d_sic-b4c-a-reduced.npy', tomo_reduced)

# ==============================================================================

# %% FOR LOOP FOR DETERMINING CENTER
# ==============================================================================
center_list = np.arange(20, 30, 1)  # LIST OF PIXEL VALUES FOR CENTER

# %%
reconstruction_matrix = np.zeros(
    [10, reconstruction_fbp.shape[0], reconstruction_fbp.shape[1]])

for i in range(0, 10):
    center = center_list[i]*effective_pixel
    # def tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=1024,start_tomo_ang=0., end_tomo_ang=360.,tomo_num_imgs=360, center=0.,pixel_size=0.00148):
    sinogram = np.squeeze(stack[:, layer_row, :])
    rotation_axis_pos = -int(np.round(center/effective_pixel))
    # rotation_axis_pos=13
    theta = np.linspace(start_tomo_ang, end_tomo_ang,
                        tomo_num_imgs, endpoint=False)
    max_rad = int(cross_sectional_dim/effective_pixel/2. *
                  1.1)  # 10% slack to avoid edge effects

    if rotation_axis_pos >= 0:
        sinogram_cut = sinogram[:, 2*rotation_axis_pos:]
    else:
        sinogram_cut = sinogram[:, :(2*rotation_axis_pos)]
    dist_from_edge = np.round(sinogram_cut.shape[1]/2.).astype(int)-max_rad

    sinogram_cut = sinogram_cut[:, dist_from_edge:-dist_from_edge]

    print('Inverting Sinogram....')
    reconstruction_fbp = iradon(sinogram_cut.T, theta=theta, circle=True)
    #plt.imshow(reconstruction_fbp, vmin=0.0, vmax=0.005)
    reconstruction_matrix[i, :, :] = reconstruction_fbp

# %%
layer_no = 7
plt.figure('check_tomo')
#plt.imshow(tomo_3d[layer_no,:,:],vmin = 0.0, vmax = 0.001)
#plt.imshow(recon_clean[0,600:2100,600:2100], vmin= 0.0001, vmax=0.0005, cmap='viridis')
plt.imshow(filters.gaussian(
    reconstruction_matrix[layer_no, :, :], 0.2), vmin=0.0005, vmax=0.002)
#######################################
# %%
##########SLOW TOMO HERE to BOTTOM#############

sinograms = np.zeros(
    [1, rad_stack.shape[0], rad_stack.shape[1], rad_stack.shape[2]])
sinograms[0, :, :] = rad_stack
sinograms = np.swapaxes(sinograms, 1, 2)
# rad_stack_0[0,:,:,:]
#rad_stack = np.swapaxes(rad_stack,0,1)

#sinograms = rad_stack_0
imageBounds = [0, stack.shape[2]]  # img_y_bounds
layers = [0, stack.shape[1]]  # img_x_bounds

topVals = tf.launchValHelper(sinograms, imageBounds, layers[0],
                             layers, theta, sigma=.4, ncore=24, vmin=-0.0001, vmax=.0005)
# %%
topCenter = 10
bottomCenter = 10
layers = np.array([600, 601])
centers = tf.calcCenters(layers, topCenter, bottomCenter)
secondary_iterations = 50

print('Reconstructing')
recon_clean = tf.reconstruct(sinograms, centers, imageBounds, layers,
                             theta, ncore=24, algorithm='gridrec', run_secondary_sirt=True, secondary_iter=secondary_iterations)
# %%
topCenter = 10
bottomCenter = 10
layers = np.array([600, 601])
secondary_iterations = 100
centers = tf.calcCenters(layers, topCenter, bottomCenter)
reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
                                theta, ncore=24, algorithm='gridrec', run_secondary_sirt=True, secondary_iter=secondary_iterations)
