#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:19:26 2019

@author: ken38
"""
#%%
import numpy as np
from pyevtk.hl import gridToVTK
#import matplotlib.pyplot as plt
#from scipy.ndimage import zoom
#from skimage.restoration import denoise_tv_chambolle
#import scipy.ndimage as img
#%%

data_list = ['recon_top_1_55.npy', 'recon_mid_1_55.npy', 'recon_bot_1_55.npy']#, 'recon_bot_2.npy']
#data_list = ['recon_bot_1_55.npy']
data_save = ['dn_top_1_55.npy', 'dn_mid_1_55.npy', 'dn_bot_1_55.npy']#, 'dn_bot_2a.npy']
#data_save = ['dn_bot_2a.npy']

#%%
data=np.load(data_list[1])
data = data[0,:,200:1500,1100:1800]

#%%
layer = 700
gauss = img.gaussian_filter(data,sigma=0.5)
edge= denoise_tv_chambolle(gauss[layer,:,:], weight = 0.0015)
#%%

plt.imshow(edge[:,:],vmin=0.0005, vmax=0.002)
plt.imshow(edge>0.0017,alpha=0.7)
#%%
for i in range (0,len(data_list)):
    print data_list[i]
    data = np.load(data_list[i]).astype('float32')
    data = data[0,:,200:1500,1100:1800]
    #edge = np.zeros([data.shape[0],data.shape[1],data.shape[2]])
    gauss = img.gaussian_filter(data,sigma=0.5)
    edge = denoise_tv_chambolle(gauss, weight = 0.0015)
    #for ii in range(0,data.shape[0]):
    #for ii in range(5,6):        
    #    print ii
    #    edge[ii,:,:] = denoise_tv_chambolle(data[ii,:,:], weight = 0.0015)
    np.save(data_save[i],edge)
    print data_save[i]

#data1 = np.load('recon_top_1.npy').astype('float32')#top
#data2 = np.load('recon_mid_1.npy').astype('float32')#middle
#data3 = np.load('recon_bot_1.npy').astype('float32')#bottom
#data4 = np.load('recon_bot_2.npy').astype('float32')#bottom2
    
-#%%
dn4 = np.load(data_save[0]).astype('float32')

#%%
dn1 = np.load(data_save[0]).astype('float32')
dn2 = np.load(data_save[1]).astype('float32')
dn3 = np.load(data_save[2]).astype('float32')
#dn4 = np.load(data_save[3]).astype('float32')

#%%
d1bi = (dn1>0.0015).astype('float32')
d2bi = (dn2>0.0015).astype('float32')
d3bi = (dn3>0.0015).astype('float32')
#d4bi = (dn4>0.0008).astype('float32')

#%%
layer = 500
#d2bi = (dn2>0.0007).astype('float32')
plt.close('all')
plt.figure('binaryimage')
plt.imshow(dn2[layer])
plt.imshow(d2bi[layer], alpha = 0.3)

#edge1 = denoise_tv_chambolle(data[700,:,:],weight=0.0012)
#%%
binary_1um = np.zeros([d1bi.shape[0]+d2bi.shape[0]+d3bi.shape[0],d1bi.shape[1],d1bi.shape[2]])
binary_1um[0:d1bi.shape[0],:,:] = d1bi
binary_1um[d1bi.shape[0]:d1bi.shape[0]+d2bi.shape[0],:,:] = d2bi
binary_1um[d1bi.shape[0]+d2bi.shape[0]:d1bi.shape[0]+d2bi.shape[0]+d3bi.shape[0],:,:] = d3bi
#binary_1um[d1bi.shape[0]+d2bi.shape[0]+d3bi.shape[0]:d1bi.shape[0]+d2bi.shape[0]+d3bi.shape[0]+d4bi.shape[0],:,:] = d4bi


#%%
stitch_nb = np.zeros([dn1.shape[0]+dn2.shape[0]+dn3.shape[0]+dn4.shape[0],dn1.shape[1],dn1.shape[2]])
    
stitch_nb[0:dn1.shape[0],:,:] = dn1
stitch_nb[dn1.shape[0]:dn1.shape[0]+dn2.shape[0],:,:] = dn2
stitch_nb[dn1.shape[0]+dn2.shape[0]:dn1.shape[0]+dn2.shape[0]+dn3.shape[0],:,:] = dn3
stitch_nb[dn1.shape[0]+dn2.shape[0]+dn3.shape[0]:dn1.shape[0]+dn2.shape[0]+dn3.shape[0]+dn4.shape[0],:,:] = dn4


#%%
stitched_data = np.load('binary_stitched_55.npy').astype('float32')

x_dim = stitched_data.shape[1] 
y_dim = stitched_data.shape[0]
z_dim = stitched_data.shape[2]
gridToVTK("stitched_binary_full_55b", np.arange(y_dim), np.arange(x_dim), np.arange(z_dim), pointData = {'data':stitched_data})

#%%
np.save('binary_stitched_55.npy',binary_1um)

#%%
data = np.load('binary_stitched_55.npy').astype('int8')

where = np.where(data==0)
#%%
nandata[where[0],where[1],where[2]] = NaN
#%%
x_dim = data.shape[1] 
y_dim = data.shape[0]
z_dim = data.shape[2]
gridToVTK("tomo_bi_55_py4_i8", np.arange(y_dim), np.arange(x_dim), np.arange(z_dim), pointData = {'data': data})

#%%
plt.imshow(data[2500,:,:])