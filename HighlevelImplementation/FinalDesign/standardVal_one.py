# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:53:00 2020

@author: User

This file stores the parameters needed for the design
"""


class standardVal:
    scale = 1
    t= 150
    t_start = -20
    t_end = 20
    
    pixel_size = 28
    
    imageSize = pixel_size * pixel_size #Total number of input neurons
    
    n = 3 #number of neurons in layer 2
    
    pth = scale * 50
    Pref = 0
    Prest = 0
    Pmin = -5.0 * scale
    
    epoch = 20
    