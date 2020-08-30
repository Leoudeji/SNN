# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:41:39 2020

@author: ludej

This script calculates the threshold for an image based on its spiking activity
"""
import random
import os
import numpy as np
from matplotlib import pyplot as plt
from fixedVal_one import fixedVal as par
from neuron_two import neuron
from receptive_field_four import rf
from spike_train_five import encode 
from reconstruct_eight import reconst_weights
from learning_rule_six import rl
from learning_rule_six import update


def threshold(train):
    tu = np.shape(train[0])[0]
    thresh = 0
    for i in range(tu):
        simul_active = sum(train[:,i])
        if simul_active>thresh:
            thresh = simul_active
    return (thresh/3)*par.scale
    
    
if __name__ == '__main__':
    img = np.array(Image.open("mnist1/" + str(1) + ".png", 0))
    print(img)
    #pot = rf(img)
    #train = np.array(encode(pot))
    #print threshold(train)



