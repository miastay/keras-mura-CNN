#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:54:53 2019

@author: ryan
"""


def alter_image(path) :
    from skimage.io import imread
    import numpy as np
    old_img = np.asarray(imread(path, as_gray = True))
    if(old_img.dtype != "uint8") :
        old_img *= 255
    print(old_img.dtype)
    if len(old_img[0]) < 499 :
            old_img = np.reshape(old_img, (512, old_img.shape[1], 1))
            index = int((512-len(old_img[0]))/2) +1
            end_index = 512-index
            img = np.full((512,512,1), 0)
            for r in range(512) :
                for c in range(index, end_index) :
                    img.itemset((r,c,0), old_img.item((r,c-index,0)))
    #first, identify background shade
    #this should be the minimum that's greater than 0
    minimum = 255
    for r in range(512) :
        for c in range(img.shape[1]) :
            if(img.item(r,c,0) < minimum and img.item(r,c,0) > 0) :
                minimum = img.item(r,c,0)
    print(minimum)
    s = 4
    for r in range(512-s) :
        for c in range(img.shape[1]-s) :
            sum = 0
            numFull = 0
            for rsub in range(s) :
                for csub in range(s) :
                    sum += img.item(r+rsub, c+csub, 0)
                    if(img.item(r+rsub, c+csub, 0) == 255) :
                        numFull += 1
            if(sum/16 > 150 and numFull > 5) :
                for rsub in range(s) :
                    for csub in range(s) :
                        img.itemset((r+rsub, c+csub, 0), minimum)
            if(img.item(r,c,0) >= 250) :
                img.itemset((r,c,0), minimum)
                    
                
    return img
    

from matplotlib import pyplot as plot
import matplotlib.cm as cm
plot.imshow(alter_image("./networkdata/testimg3.png")[:,:,0], cmap=cm.gray)
        