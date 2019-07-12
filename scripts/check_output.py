#!/usr/bin/env python
# coding: utf-8

import h5py as h5
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re


matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = None
matplotlib.rcParams['image.cmap'] = 'jet'
np.seterr(invalid='ignore', divide='ignore')


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


def showmax(data, filename):
    sx, sy, sz = data.shape
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.log(np.nanmax(np.abs(data), axis=2)))
    plt.ylabel("x")
    plt.xlabel("y")
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.nanmax(np.abs(data), axis=1).T))
    plt.ylabel("z")
    plt.xlabel("x")
    plt.subplot(1, 3, 3)
    plt.imshow(np.log(np.nanmax(np.abs(data), axis=0)))
    plt.ylabel("y")
    plt.xlabel("z")
    plt.savefig(filename)


with h5.File('amplitude-0.hdf5','r') as f:
    acc = f['accumulator'][:]
    weight = f['weight'][:]


show(acc, "acc.png")
show(weight, "weight.png")
show(acc/weight, "amp.png")


comb = (acc + acc[::-1,::-1,::-1])/(weight + weight[::-1,::-1,::-1])
show(comb, "comb.png")
showmax(comb, "maxcomb.png")


pattern = re.compile("^rho-(\\d+)\\.hdf5$")
for filename in os.listdir():
    match = pattern.match(filename)
    if not match:
        continue
    rank = match.group(1)

    with h5.File(filename,'r') as f:
        rho = f['rho'][:]

    showmax(rho, f"rho-{rank}.png")
