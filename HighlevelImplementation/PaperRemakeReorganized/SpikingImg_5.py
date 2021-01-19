# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:24:31 2021

@author: User
"""

import numpy as np


 
#6 - Implement the Spiking signal in pg 11 of 44 formula and make the raster plot of it against pixels (figure 12)
#Keep the bins spaced at 10bins (10msecond) (gap = 2millisecond)
def spike_img(img):
    G = params.GAMMA
    pixel_H, pixel_W = img.shape #get dimension of input image
    spike_img = np.zeros((img.shape)) #declare a 2D matrix of zeros, of thesame shape as input image
    
    for i in range(pixel_H): #Loop over image height
        for j in range(pixel_W): #Loop over image width
            if ((img[i][j])) > G: #check if particular pixel has a value greater than threshold value of 50
                #spike_img[i][j] = img[i][j]
                spike_img[i][j] = 1 #assign a value of 1 to spike_img matrix at position i,j if value is greater than 50
              
    
    return spike_img



#Converts our image into spike trains (delays - in milliseconds)
#Delay equation (tau) is in Pg11/44
def spike_train(img):
    #G = params.GAMMA 
    
    pixel_H, pixel_W = img.shape
    delay = np.zeros((img.shape))
    #delay_list = []
    
    for i in range(pixel_H):
        for j in range(pixel_W):
            #if ((img[i][j])) > G:
            delay[i][j] = (1 / img[i][j]) * 1000
                #delay_list.append((img[i][j]).astype(int))
    
    
    return delay


#Takes result of the spike_train() function and create a delay list
def spike_plot(delay):
    yp = np.ceil(delay)
    delay_list = []
    T = 20
    
    for t in range(T):
        #temp = yp.copy().astype(int)
        #temp[temp != t] = 0
        #temp[temp == t] = 1
        
        temp = np.zeros_like(yp)
        
        
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if yp[i][j]==t:
                    temp[i][j] = 1
        
        
        delay_list.append(temp)
        
    return delay_list



#Plot spike train (Figure 12 - Page 11/44)    
def plot_spike_train(img_list):
    plt.figure()
    for i in range(len(img_list)):
        img = np.where(img_list[i].flatten()==1)[0]
        try:
            if len(img) > 0:
                plt.plot([i]*len(img), img, 'p', c='b')
        except:
            print(img)