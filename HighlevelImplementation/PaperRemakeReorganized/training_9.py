# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:34:48 2021

@author: User
"""
import time as T
#import necessary libraries
import imageio
import matplotlib.pyplot as plt 
import numpy as np
from params_1 import params
#from neuron_2 improt neuron
import pdb # used to debug program
from scipy import signal #Used to verify 3d convolution (test_Conv_3x3)

def SNN():
    
   result = 2* 2;
    
   return result



if __name__ == "__main__":
#def run():
    
    start = T.time()*1000 #Global time - time() function returns time in seconds
    
    print ("Start time:", start)
    
    img = imageio.imread("2.png")
    #print(img)
    
    
    d=SNN.weights()
    
    
    #The next three lines of code implements the DoG Filter on Page 9, and Figure.8 on Page 10
    on_center_filter = SNN.plot_DOG(1,2) 
    off_center_filter = SNN.plot_DOG(2,1)
    SNN.plot_on_off_filter(on_center_filter, off_center_filter) #Plots the combined ON and OFF filter in Figure 8
    
    
    
    #Convolve with the on and off filter
    #Add Cmake grey for grey image - plt.imshow(img, cmap='gray')
    plt.figure()
    img_on = SNN.convolution(img, on_center_filter)
    plt.imshow((img_on ),cmap='gray')
    
    plt.figure()
    img_off = SNN.convolution(img, off_center_filter)
    plt.imshow((img_off),cmap='gray')
    
    '''
    comb = SNN.Weights_images(img,d)
    #print(comb)
    plt.imshow(comb)
    '''
    
    
    #Spiking image
    
    #spike_on
    plt.figure()
    spi_on = SNN.spike_img(img_on)
    plt.imshow(spi_on,cmap='gray')
    
    #spike_off
    plt.figure()
    spi_off = SNN.spike_img(img_off)
    plt.imshow(spi_off,cmap='gray')
    
    
    '''
    #Spike delay (train)
    
    plt.figure()
    spi_tri_on = SNN.spike_train(img_on)
    plt.imshow(spi_tri_on,cmap='gray')
    
    
    plt.figure()
    spi_tri_off = SNN.spike_train(img_off)
    plt.imshow(spi_tri_off,cmap='gray')
    '''
    
    '''
    #Raster one - figure 3
    SNN.raster_one(img)
    '''
    
    
    #RAster 2 - figure 12: Plot of spike image against the delay
    plt.figure()
    
    
    plt.show
    #plt.eventplot(sp_plot)
    spi_train_on = SNN.spike_plot(SNN.spike_train(img_on))
    spi_train_off = SNN.spike_plot(SNN.spike_train(img_off))
    SNN.plot_spike_train(spi_train_on)
    
    plt.title('Spike raster plot') # Provide the title for the spike raster plot
    plt.xlabel('Time') #x axis fo the spike raster plot
    plt.ylabel('Pixel/Neuron')  #Y axis label for the spike raster plot
    
    
    
    '''   
    # Draw a spike raster plot
    plt.figure()
    plt.eventplot(SNN.spike_train(img),neuralData)
        
    #display the spike raster plot
    plt.show()
    '''
    
    
    
    #3D Convolution without Inhibition
    conv_on_center_filter = SNN.new_plot_DOG(1,2)
    conv_off_center_filter = SNN.new_plot_DOG(2,1)
    
    #create stacked on/off images for each time - should remove any time that has no spikes
    spi_stacked = [np.stack([on,off], axis=2) for (on,off) in zip(spi_train_on, spi_train_off)]
    
    w = SNN.new_plot_DOG(dim = 5, chanls = 2, nlayers = 30)
    
    #cImg = SNN.convolution_3x3(SNN.spike_train(img), conv_on_center_filter)
    cImg = []
    cImg = SNN.convolution_3x3(spi_stacked, w)
    
    
    #Calculate cumulative sum after 3D convoltuion
    cumSum = SNN.cumSum(spi_stacked, w, cImg)
    
    
    
    #Transform cumSum into a 2D image
    fImage =  SNN.threeDImage(cumSum)
    
    #Plot the final image
    plt.figure()
    plt.imshow(fImage)
    #plt.imshow(plt.colorbar(fImage))
    plt.figure()
    plt.colorbar(plt.pcolormesh(fImage))
    #plt.colorbar(fImage)
    
    
    
    #Test 3D convolvution
    TestCImg = SNN.test_Conv_3x3(spi_stacked, w)
    #TestCSUm = SNN.test3DCSum(TestCImg)  #The problem was with this functions
    #testshow = SNN.test3DCSum(TestCImg)
    cumsumtst = SNN.cumSum(spi_stacked, w, TestCImg)
    testshow = SNN.threeDImage(cumsumtst)
    
    '''
    plt.figure()
    plt.imshow(TestCSUm)
    plt.figure()
    plt.colorbar(plt.pcolormesh(TestCSUm))
    '''
    plt.figure()
    plt.imshow(testshow)
    
    
    
    #Image with Lateral Inhibition (3D)
    LIimage = SNN.Linhibit(cumSum)
    plt.figure()
    plt.imshow(LIimage)
    plt.figure()
    plt.colorbar(plt.pcolormesh(LIimage))
    
    
    
    #Apply Lateral inhibition to Test 3D convolvution image
    #TestLIimage = SNN.Linhibit(TestCImg)
    TestLIimage = SNN.Linhibit(cumsumtst)
    plt.figure()
    plt.imshow(TestLIimage)
    plt.figure()
    plt.colorbar(plt.pcolormesh(TestLIimage))
    
    
    #STDP Competition
    SK = SNN.STDP_Kernel(11,11)
    
    STDP_CompImg = SNN.STDP_Comp(cumSum)
    #plt.figure()
    #plt.imshow(STDP_CompImg)
    #plt.colorbar(plt.pcolormesh(LIimage))
    
    
    '''
    plt.figure()
    #plt.imshow(cImg[:,:,:])  #Will need to work on this
    #plt.colorbar()
    #plt.plot()
    plt.show(cImg)
    '''
    
   
    
    
    
    
    print ("total processing time in milliseconds: ", ((T.time()*1000) - start))
    