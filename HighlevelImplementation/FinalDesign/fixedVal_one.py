# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:53:00 2020

@author: User

This file stores the parameters needed for the design
"""


class fixedVal:
    scale = 1
    T= 150
    t_start = -20
    t_end = 20
    
    pixel_size = 28 #value can be varied depending on image dimension
    m = pixel_size * pixel_size #Total number of input neurons (layer one)
    
    n = 3 #number of neurons in layer 2
    
    Pmin = -5.0 * scale
    Pth = scale * 50
    Pref = 0
    Prest = 0
    D = 0.75 * scale
    
    w_max = 2.0 * scale
    w_min = -1.2 * scale
    sigma = 0.02
    A_minus = 0.3  #when time diffence is negative 
    A_plus = 0.8    #when time difference is positive
    tau_minus = 10
    tau_plus = 10
    epoch = 20
    
    fr_bits = 12
    int_bits = 12
    