# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:22:30 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt 
 
#5 - Convert MNIST images to spikes, using Difference of Gaussian Filter (DoG filter) (pg.9 of 44)
# DoG Filter imitates the retinal receptive field of the human eye

#Formula is provided on Page 9
#(ON center spike is produced if pixel value increases and OFF center spike is produced if pixel value decrease - page 9 of 44)
#notice that the value of sig1 and sig2 changes depending on whether it's ON or OFF center
def plot_DOG(sigma1, sigma2):
    #Fix scale - Problem 
   
    dim = 7 #2D filter dimension
    
    total = np.zeros((dim,dim)) #create/declare a 5*5 matrix of zeros
    
    for i in range(dim):   #Leo - 5 x 5 filter size comes from page 10 and 17, of 44
        for j in range(dim):
            
            #Equation of filter on Pg 9of 44 is lengthy. We break it in two below (frac1 and frac2)
            frac1 = 1/(2 * np.pi * sigma1**2)
            frac2 = 1/(2 * np.pi * sigma2**2)
            
            expp = ((i-3)**2)  + ((j-3)**2)
            
            #if((-3<=i and i>=3) and (-3<=j and j>=3) )
            total[i][j] = (frac1 * np.exp(-expp/(2*sigma1**2))) - (frac2 * np.exp(-expp/(2*sigma2**2))) #Full implementation of Equation 'K' in Page 9 of 44
            
            
    #Added lines normalizes the filter before using it for convolution
    total = total - np.mean(total)
    total = total / np.max(np.abs(total))
            
            
    return total


#Function below plots and ON and OFF filters
def plot_on_off_filter(on_center_filter, off_center_filter):
    plt.figure()
    plt.colorbar(plt.pcolormesh(on_center_filter))


    plt.figure()
    plt.colorbar(plt.pcolormesh(off_center_filter))
    
  
    
#Multiply weights and images
def Weights_images(img,weight):
    result = np.dot(img,weight)
    return result



#9 - Function defines New Kernel/Filter in Figure 13 (Page 12/44)
def new_plot_DOG(dim = 5, chanls = 2, nlayers = 30):
   #Fix scale - Problem 
    
    #dim_total = np.zeros((dim,dim))
    #ctotal = np.zeros((dim,dim,chanls))
    
    
    '''
    for p in range(chanls): #Lopp over channels
        for i in range(dim):   #Leo - 5 x 5 filter size comes from page 5 of 44
            for j in range(dim):
    '''
    
    ctotal = np.random.normal(0.8, 0.01, size=(dim,dim, chanls, nlayers))  #I got this normal distfrom page 22
    #ctotal = np.random.normal(0.8, 0.01, size=(dim,dim, chanls))  #I got this normal distfrom page 22
            

    #Added lines
    #ctotal = ctotal- np.mean(ctotal)
    #ctotal = ctotal / np.max(ctotal)
   
    
    
    return ctotal