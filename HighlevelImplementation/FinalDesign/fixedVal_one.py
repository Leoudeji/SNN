# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:53:00 2020

@author: User

This file stores the parameters needed for the design
"""


class fixedVal:
    scale = 1
    T= 150  #Leo- Duraton in which sample is presented to network
    t_start = -20
    t_end = 20
    
    #pixel_size = 16 #value can be varied depending on image dimension. It's very important for classification
    pixel_size = 28 #value can be varied depending on image dimension
    m = pixel_size * pixel_size #Total number of input neurons (layer one)
    
    #n = 3 #number of neurons in layer 2
    n = 10 #number of neurons in layer 2 #Changed today Sept 29
    #'n' must be greater than or equal to "num_of_images" below.
    #In classification i.e. "classify_ten.py" tthis represents the number of bits used to encode/decode the images
    
    Pmin = -5.0 * scale
    #Pth = scale * 50
    
    
    Pth = scale * 6 #Chnaged today Sept 29 
    #The value f Pth significantly alters prediction accuracy
    
    
    Pref = 0
    Prest = 0
    D = 0.75 * scale  #Leo - Recovery variable
    
    w_max = 2.0 * scale  #Leo-weight initialization
    w_min = -1.2 * scale
    sigma = 0.02  #Leo-Learning rate
    A_minus = 0.3  #when time diffence is negative 
    A_plus = 0.8    #when time difference is positive
    tau_minus = 10
    tau_plus = 10
    epoch = 35  #20
    
    fr_bits = 12
    int_bits = 12
    
    num_of_images = 10  #Defined today Sept 29
    