#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:19:26 2019

@author: ken38
"""
#%%
import numpy as np
from pyevtk.hl import gridToVTK
import matplotlib.pyplot as plt
#from scipy.ndimage import zoom
import scipy.ndimage.morphology as morphology
import ctypes

#%% multislice viewer here #%%

#def previous_slice():
#    pass
#
#def next_slice():
#    pass

#plt.rcParams['keymap.<command>'] = ['<key1>', '<key2>']
def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('Layer %d' % (ax.index))

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title('Layer %d' % (ax.index))
    
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index],interpolation='none')
    ax.set_title('Layer %d' % (ax.index))
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
   # if event.key == 

#%%
data1 = np.load('recon_top_1.npy').astype('float32')#top
#data2 = np.load('recon_mid_1.npy').astype('float32')#middle
#data3 = np.load('recon_bot_1.npy').astype('float32')#bottom
#data4 = np.load('recon_bot_2.npy').astype('float32')#bottom2
#%%
data1 = data1[0,:,200:2000,1200:2400]

#plt.imshow(data1[0,800,400:2000,1200:2400])


#%% laplace of gaussian blob detector
from skimage.restoration import denoise_tv_chambolle

edge1 = np.zeros([data1.shape[0],data1.shape[1],data1.shape[2]])

for i in range(0,10):
    print i
    edge1[i*10:data1.shape[1]*i*10+10,:,:] = denoise_tv_chambolle(data1[i*10:data1.shape[1]*i*10+10,:,:], weight = 0.001)
#edges2 = denoise_tv_chambolle(data2, weight = 0.001)
#edges3 = denoise_tv_chambolle(data3, weight = 0.001)
#edges4 = denoise_tv_chambolle(data4, weight = 0.001)
#%%
binary1 = (edges1>0.0016).astype('float32')
binary2 = (edges2>0.0016).astype('float32')
binary3 = (edges3>0.0016).astype('float32')

np.save('binary1.npy', binary1)
np.save('binary2.npy', binary2)
np.save('binary3.npy', binary3)
#%%
binary1 = np.load('binary1.npy')
binary2 = np.load('binary2.npy')
binary3 = np.load('binary3.npy')

#%%

layer1 = 725+50
layer2 = 0+50
offsetx = 1
offsety = 1

plt.close('all')
plt.figure('overlap')
#plt.imshow(binary1[layer1,:,:],alpha = 0.2)
plt.imshow(binary2[layer2,:,:], alpha = 0.2, cmap = 'bone')
plt.imshow(binary_new[layer1,:,:], alpha = 0.2)

#%% stupid way to shift but will work

x = 6
y = 5

binary_mid = np.zeros([binary1.shape[0],binary1.shape[1]+x, binary1.shape[2]+y])
binary_mid[:,x:,:binary_mid.shape[2]-y] = binary1

binary_new = np.zeros([binary1.shape[0],binary2.shape[1],binary2.shape[2]])
binary_new[:,:,:] = binary_mid[:,:binary_mid.shape[1]-x,:binary_mid.shape[2]-y]


layer1 = 725+50
layer2 = 0+50
offsetx = 1
offsety = 1

plt.close('all')
plt.figure('overlap')
#plt.imshow(binary1[layer1,:,:],alpha = 0.2)
plt.imshow(binary2[layer2,:,:], alpha = 0.2, cmap = 'bone')
plt.imshow(binary_new[layer1,:,:], alpha = 0.2)



#%%
offset = 50
stitched_data = np.zeros([725*2+927-offset,binary1.shape[1],binary2.shape[2]])

#%%
offset = 50
stitched_data[0:725,:,:] = binary1[offset:offset+725]
stitched_data[725:725*2,:,:] = binary2[offset:offset+725]
stitched_data[725*2:725*2+927-offset,:,:] = binary3[offset:]
#%%
stitched_data = stitched_data.astype('float32')

#%%
layer = 725
plt.close('all')
plt.figure()
plt.imshow(stitched_data[layer,:,:])

#%%
x_dim = stitched_data.shape[1] 
y_dim = stitched_data.shape[0]
z_dim = stitched_data.shape[2]
gridToVTK("stitched_binary_full", np.arange(y_dim), np.arange(x_dim), np.arange(z_dim), pointData = {'data': stitched_data})

#%%
binary_version = 

x_dim = data.shape[1] 
y_dim = data.shape[0]
z_dim = data.shape[2]
gridToVTK("tomo_bi_1", np.arange(y_dim), np.arange(x_dim), np.arange(z_dim), pointData = {'data': binary_version})
