# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:34:31 2020

@author: ludej
Receptors are defined here
The receptors perform convolution of the input image and the receptive field, to make reconstruction
of the original image visible
"""

import numpy as np
import imageio
from fixedVal_one import fixedVal as par

def rf(inp):
    sca1 = 0.625
    sca2 = 0.125
    sca3 = -0.125
    sca4 = -0.5
    
    #Kernel for the receptive field
    w = [[ sca4, sca3, sca2, sca3, sca4],
        [  sca3, sca2, sca1, sca2, sca3],
        [  sca2, sca2,    1, sca1, sca2],
        [  sca3, sca2, sca1, sca2, sca3],
        [  sca4, sca3, sca2, sca3, sca4]]
        
    #pot = np.zeros([par.pixel_size,par.pixel_size])  #Leo: Changed
    pot = np.zeros([inp.shape[0],inp.shape[1]])
    ran = [-2,-1,0,1,2]
    ox = 2
    oy = 2
    
    #Perform Convolution Operation (changed)
    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            summ = 0
            for m in ran:
                for n in ran:
                    if (i+m)>=0 and (i+m)<=inp.shape[0]-1 and (j+n)>=0 and (j+n)<=inp.shape[0]-1:
                        summ = summ + w[ox+m][oy+n] * inp[i+m][j+n]/255
            pot[i][j] = summ
    return pot
#    for i in range(par.pixel_size):
#        for j in range(par.pixel_size):
#            summ = 0
#            for m in ran:
#                for n in ran:
#                    if (i+m)>=0 and (i+m)<=par.pixel_size-1 and (j+n)>=0 and (j+n)<=par.pixel_size-1:
#                        summ = summ + w[ox+m][oy+n]*inp[i+m][j+n]/255
#            pot[i][j] = summ
#    return pot
        
if __name__ == '__main__':
    img = imageio.imread("images/" + str(100) + ".png")
    pot = rf(img)
    max_a = []
    min_a = []
    for i in pot:
        max_a.append(max(i))
        min_a.append(min(i))
    for i in range(16):
        temp = ''
        for j in pot[i]:
            temp += '%02d ' % int(j)
        print(temp)
    print("max", max(max_a))
    print("min", min(min_a))