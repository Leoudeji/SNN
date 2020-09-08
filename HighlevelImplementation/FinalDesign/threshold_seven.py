# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:41:39 2020

@author: ludej

This script calculates the threshold for an image based on its spiking activity
"""

import numpy as np
from fixedVal_one import fixedVal as par


def threshold(train):
    tu = np.shape(train[0])[0]
    thresh = 0
    for i in range(tu):
        simul_active = sum(train[:,i])
        if simul_active>thresh:
            thresh = simul_active
    return (thresh/3)*par.scale
    
    
#if __name__ == '__main__':
#    img = np.array(Image.open("mnist1/" + str(1) + ".png", 0))
#    print(img)
    #pot = rf(img)
    #train = np.array(encode(pot))
    #print threshold(train)



