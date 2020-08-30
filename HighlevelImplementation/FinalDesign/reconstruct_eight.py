# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:42:25 2020

@author: ludej

This script helps us reconstruct the images to evaluate the training process. 
The generative property of the spiking neural network
make this possible. The function 'reconst_weights' is for this purpose
"""

import imageio
import numpy as np
from numpy import interp
from receptive_field_four import rf
from fixedVal_one import fixedVal as par


def reconst_weights(weights, num):
    weights = np.array(weights)
    weights = np.reshape(weights, (par.pixel_size,par.pixel_size))
    img = np.zeros((par.pixel_size,par.pixel_size))
    for i in range(par.pixel_size):
        for j in range(par.pixel_size):
            img[i][j] = int(interp(weights[i][j], [par.w_min,par.w_max], [0,255]))

    imageio.imwrite('neuron' + str(num) + '.png', img)
    return img
    
    
def reconst_rf(weights, num):
	weights = np.array(weights)
	weights = np.reshape(weights, (par.pixel_size,par.pixel_size))
	img = np.zeros((par.pixel_size,par.pixel_size))
	for i in range(par.pixel_size):
		for j in range(par.pixel_size):
			img[i][j] = int(interp(weights[i][j], [-2,3.625], [0,255]))

	imageio.imwrite('neuron' + str(num) + '.png', img)
	return img
    
 
if __name__ == '__main__':
    img = imageio.imread("images2/" + "69" + ".png")
    pot = rf(img)
    reconst_rf(pot, 12)