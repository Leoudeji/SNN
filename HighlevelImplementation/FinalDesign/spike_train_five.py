# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:40:26 2020

@author: ludej

Generate spike train based on rate of change in the potential map
"""

import math
import imageio
import numpy as np
from numpy import interp
from fixedVal_one import fixedVal as par
from receptive_field_four import rf

#(changed)
def encode(pot):
    
    #initialize spike train
    train = []
    '''
    for l in range(pot.shape[0]):
        for m in range(pot.shape[1]):
    '''
    
    for l in range(28):
        for m in range(28):

            temp = np.zeros([(par.T+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pot[l][m], [-1.069,2.781], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pot[l][m]>0):
                while k<(par.T+1):
                    temp[int(k)] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train
    
    
    


def encode2(pixels):

    
    #initializing spike train
    train = []
    '''
    for l in range(pixels.shape[0]):
        for m in range(pixels.shape[1]):
    '''
    
    for l in range(28):
        for m in range(28):

            temp = np.zeros([(par.T+1),])

            #calculating firing rate proportional to the membrane potential
            freq = interp(pixels[l][m], [0, 255], [1,20])
            #print(pot[l][m], freq)
            # print freq

            assert freq > 0

            freq1 = math.ceil(600/freq)

            #generating spikes according to the firing rate
            k = freq1
            if(pixels[l][m]>0):
                while k<(par.T+1):
                    temp[k] = 1
                    k = k + freq1
            train.append(temp)
            # print sum(temp)
    return train
    
    
    
if __name__  == '__main__':
        m = []  #This change happened today (tis line and next
        n = []
   
        img = imageio.imread("/Users/ludej/OneDrive/Desktop/Spring2020/Research/SummerWork/Week12_August24/HighlevelImplementation/SNN/HighlevelImplementation/FinalDesign/training/0.png")
   

        pot = rf(img)

    # for i in pot:
    #     m.append(max(i))
    #     n.append(min(i))

    # print max(m), min(n)
    #train = encode2(img)
        train = encode(pot)
        f = open('train6.txt', 'w')
        print(np.shape(train))

        for j in range(len(train)):
            for i in range(len(train[j])):
                f.write(str(int(train[j][i])))
                f.write('\n')

        f.close()

