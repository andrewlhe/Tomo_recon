#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 07:43:07 2020

@author: ken38
"""

#%% Necessary Dependencies
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
#% DETECTOR PARAMETERS
#==============================================================================
nrows = 3008
ncols = 4112
pixel_size = 0.00345/5

#SCAN PARAMETERS
start_tomo_ang = 180.0
end_tomo_ang= 0.0
tomo_num_imgs = 720

#%%============================================================================

stack1 = np.load('red_stack_55p_top.npy')#np.load('red_stack_z_top.npy')#top
stack2 = np.load('red_stack_55p_mid.npy')#np.load('red_stack_z_mid.npy')#middle
stack3 = np.load('red_stack_55p_bot.npy')#np.load('red_stack_z_bot.npy')#bottom

#%% tomopy with SIRT
stack = stack1
sinograms = np.zeros([1,stack.shape[0],stack.shape[1],stack.shape[2]])
sinograms[0,:,:] = stack
sinograms = np.swapaxes(sinograms,1,2)
imageBounds = [0,stack.shape[2]]#img_y_bounds
theta = np.deg2rad(np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False))

#%
topCenter = -30
bottomCenter = -23
layers = np.array([56,853])
secondary_iterations = 50
centers = tf.calcCenters(layers,topCenter, bottomCenter) 
reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
                    theta, sigma = 0.1 , ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
np.save('recon_top_1_55.npy', reconstruction)
#%%
stack = stack2
sinograms = np.zeros([1,stack.shape[0],stack.shape[1],stack.shape[2]])
sinograms[0,:,:] = stack
sinograms = np.swapaxes(sinograms,1,2)
imageBounds = [0,stack.shape[2]]#img_y_bounds
theta = np.deg2rad(np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False))

topCenter = -30
bottomCenter = -23
layers = np.array([56,853])
secondary_iterations = 100
centers = tf.calcCenters(layers,topCenter, bottomCenter) 
reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
                    theta, sigma = 0.1 , ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
np.save('recon_mid_1_55.npy', reconstruction)
#%%
stack = stack3
sinograms = np.zeros([1,stack.shape[0],stack.shape[1],stack.shape[2]])
sinograms[0,:,:] = stack
sinograms = np.swapaxes(sinograms,1,2)

topCenter = -30
bottomCenter = -21.53
layers = np.array([56,1020])
secondary_iterations = 100
centers = tf.calcCenters(layers,topCenter, bottomCenter) 
reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
                    theta, sigma = 0.1 , ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
np.save('recon_bot_1_55.npy', reconstruction)
#%%
#stack = stack3
#sinograms = np.zeros([1,stack.shape[0],stack.shape[1],stack.shape[2]])
#sinograms[0,:,:] = stack
#sinograms = np.swapaxes(sinograms,1,2)
#imageBounds = [0,stack.shape[2]]#img_y_bounds
#theta = np.deg2rad(np.linspace(start_tomo_ang, end_tomo_ang, tomo_num_imgs, endpoint=False))
#
#topCenter = -29
#bottomCenter = -27.53
#layers = np.array([1070,1280])
#secondary_iterations = 50
#centers = tf.calcCenters(layers,topCenter, bottomCenter) 
#reconstruction = tf.reconstruct(sinograms, centers, imageBounds, layers,
#                    theta, sigma = 0.1 , ncore = 24,algorithm='gridrec',run_secondary_sirt=True,secondary_iter=secondary_iterations)
#np.save('recon_bot_2a.npy', reconstruction)
#%%
#layers = [70,1070]#img_x_bounds
#topVals = tf.launchValHelper(sinograms, imageBounds, layers[0],layers, theta, sigma = .4, ncore = 24)# , vmin=-0.0001, vmax=.0005)
from skimage.restoration import denoise_tv_chambolle

edges1 = denoise_tv_chambolle(reconstruction[0,1,:,:], weight = 0.001)
#%%
plt.close('all')
plt.figure()
plt.imshow(edges1[:,:], vmin=0.0004, vmax = 0.002)