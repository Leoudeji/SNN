# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:25:13 2021

@author: User
"""

import numpy as np

 
#Functons to Implement STDP competition

def STDP_Kernel(u,v):
    
    SK= np.random.normal(0, 0, size=(u,v)) 
    
    
    return SK




#Function creates results in Figure 16 (Page 14/44) - Incomplete
def STDP_Comp(cmpnd_npad_img):
    
    gam = 15
    
   
    #fh, fw = SK.shape
    inputImg = cmpnd_npad_img
    
    img_pixel1, img_pixel2, n_layers,t = np.shape(cmpnd_npad_img)
        
    stdpCompResult = np.zeros((img_pixel1, img_pixel2)) #padded image
        
    
    #LIimg3D = np.zeros((img_pixel1, img_pixel2)) #add t
    findMax = np.zeros((img_pixel1, img_pixel2))
    neuron = np.zeros((img_pixel1, img_pixel2, n_layers,t))
    
    
    #for every pixel the max should be 30
     
            
    #The difference from lateral inhibition is 'n' in findMax and position of loop over time
                    
           
    for n in range(n_layers): #Loop over all layers
               
        for Tm in range(t):
                    
            for i in range(img_pixel1):
                for j in range(img_pixel2):
            
                    findMax = max(inputImg[:,:,n_layers,t-1])
                    if ((findMax)) > gam:
                        
                        #if ((inputImg[i][j][n][Tm])) > gam:
                        
                        neuron = inputImg[i,j,n,Tm]
                        stdpCompResult = 1
                           
                           
    #Second stage of Spiking competition - sort neurons and apply (+ or - 5 limit filters)
    
    
                           
    '''
    img_pixel1, img_pixel2 = LIimg3D.shape
    fh, fw = SK.shape
        
    stdpCompResult = np.zeros((img_pixel1, img_pixel2)) #padded image
        
    for i in range(img_pixel1):
            for j in range(img_pixel2):
                
                #Run filter across image
                for k in range(fh):
                    for l in range(fw):
                        if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                            stdpCompResult[i][j] += img[i + k - fh][j+l-fw] * SK[k][l]
    '''                    
                                        
    
    
    return stdpCompResult