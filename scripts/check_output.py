#!/usr/bin/env python

# coding: utf-8

# In[1]:


import h5py as h5
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = None
matplotlib.rcParams['image.cmap'] = 'jet'


# In[3]:


def show(data, filename):
    sx, sy, sz = data.shape
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.log(np.abs(data[:,:,sz//2])))
    plt.ylabel("x")
    plt.xlabel("y")
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(data[:,sy//2,:]).T))
    plt.ylabel("z")
    plt.xlabel("x")
    plt.subplot(1, 3, 3)
    plt.imshow(np.log(np.abs(data[sx//2,:,:])))
    plt.ylabel("y")
    plt.xlabel("z")
    plt.savefig(filename)


# In[13]:


def showmax(data, filename):
    sx, sy, sz = data.shape
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.log(np.abs(np.nanmax(data, axis=2))))
    plt.ylabel("x")
    plt.xlabel("y")
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(np.nanmax(data, axis=1).T)))
    plt.ylabel("z")
    plt.xlabel("x")
    plt.subplot(1, 3, 3)
    plt.imshow(np.log(np.abs(np.nanmax(data, axis=0))))
    plt.ylabel("y")
    plt.xlabel("z")
    plt.savefig(filename)


# In[4]:


with h5.File('amplitude-0.hdf5','r') as f:
    acc = f['accumulator'][:]
    weight = f['weight'][:]
    #data = np.fft.fftshift(data)


# In[5]:


show(acc, "acc.png")
show(weight, "weight.png")


# In[6]:


show(acc/weight, "amp.png")


# In[8]:


comb = (acc + acc[::-1,::-1,::-1])/(weight + weight[::-1,::-1,::-1])
show(comb, "comb.png")


# In[14]:


showmax(comb, "maxcomb.png")

