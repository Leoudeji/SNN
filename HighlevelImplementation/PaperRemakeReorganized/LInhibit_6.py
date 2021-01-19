# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:24:54 2021

@author: User
"""
import numpy as np


#This function implements Lateral inhibition (Figure 15 - Page 13/44)
def Linhibit(cmpnd_npad_img):
    
    gam = 15
    
    inputImg = cmpnd_npad_img
    
    img_pixel1, img_pixel2, n_layers,t = np.shape(cmpnd_npad_img)
    
    
    
    LIimg3D = np.zeros((img_pixel1, img_pixel2)) #add t
    findMax = np.zeros((img_pixel1, img_pixel2))
    
    
    #for every pixel the max should be 30
     
            
    for i in range(img_pixel1):
        for j in range(img_pixel2):
                    
            for Tm in range(t):
                for n in range(n_layers): #Loop over all layers
            
                    findMax = max(inputImg[i,j,:,t-1])
                    if ((findMax)) > gam:
                        
                        #if ((inputImg[i][j][n][Tm])) > gam:
                        
                       
                            LIimg3D[i][j] = 1
                        
                            #findMax = max(inputImg[i][j][n][Tm])
                            #len(cumSum[1,1,:,1])
                            #max(cumSum[2,2,:,19])
                            
                            '''
                            #Test Algorithm
                            
                            inputImg = cumSum
                            i, j, n,Tm = np.shape(cumSum)
                            
                            for i in range (n):
                                findMax = max(inputImg[2,2,:,19])
                            
                            '''
        #print(Tm)
    
                                        
    return LIimg3D