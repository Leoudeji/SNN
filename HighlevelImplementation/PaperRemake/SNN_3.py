# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:45:19 2020

@author: ludej
"""

#SNN implementation

import imageio
import time as T
import matplotlib.pyplot as plt
import numpy as np
from params_1 import params
#from neuron_2 improt neuron


#1 - Create function to replicate figure 1
class SNN():  
    
    #2 - Create Function for Input Neuron versus spike time (Figure 3) (How did they use 100 input neurons? )
    
    
    
    
    #3 - Set random weights from normal distribution
    #draw samples from distribution
    
    def weights():
        mu, sigma = 0.5, 0.05 # mean and standard deviation - values come from page 4 of 44
        #s = np.random.normal(mu, sigma, 784)
        s = np.random.normal(mu,sigma,size=(28, 28))
        
        #Verify the mean and the variance:
        abs(mu - np.mean(s)) #may vary
        abs(sigma - np.std(s, ddof=1)) #may vary
        
        print("size of random is: ", len(s))
        
        #Display the histogram of the samples, along with the probability density function:
        count, bins, ignored = plt.hist(s, 30, density=True)
        plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                       np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                 linewidth=2, color='r')
        plt.show()
       
        '''
        #Two-by-four array of samples from N(3, 6.25):
        np.random.normal(3, 2.5, size=(2, 4))
        '''
        return s
    
    
    #4 - Convolution
    #Get code to perform convolution operation on an input image
    #We use same-mode convolution; so the shape of the ouput and the input image remains the same (page 5 of 44)
    #How do we incorporate time and thresholding into formula as in page 5??

    def convolution(img, total):
        img_pixel1, img_pixel2 = img.shape
        fh, fw = total.shape
        
        pad_img = np.zeros((img_pixel1, img_pixel2)) #padded image
        
        for i in range(img_pixel1):
            for j in range(img_pixel2):
                
                #Run filter across image
                for k in range(fh):
                    for l in range(fw):
                        if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                            pad_img[i][j] += img[i + k - fh][j+l-fw] * total[k][l]
                            
        return pad_img

    
    
    
    #5 - Convert MNIST images to spikes, using Difference of Gaussian Filter (DoG filter) (pg.9 of 44)
    # DoG Filter imitates the retinal receptive field of the human eye
    
    #Formula is provides on page - Already done by Sanjay
    #(ON center spike is produced if pixel value increases and OFF center spike is produced if pixel value decrease - page 9 of 44)
    #notice that the value of sig1 and sig2 changes depending on whether it's ON or OFF center
    
    def plot_DOG(sigma1, sigma2):
       #Fix scale - Problem 
       
        dim = 7
        
        total = np.zeros((dim,dim))
        
        for i in range(dim):   #Leo - 5 x 5 filter size comes from page 5 of 44
            for j in range(dim):
                
                frac1 = 1/(2 * np.pi * sigma1**2)
                frac2 = 1/(2 * np.pi * sigma2**2)
                
                expp = ((i-3)**2)  + ((j-3)**2)
                
                #if((-3<=i and i>=3) and (-3<=j and j>=3) )
                total[i][j] = (frac1 * np.exp(-expp/(2*sigma1**2))) - (frac2 * np.exp(-expp/(2*sigma2**2)))
                
                '''
                #Added lines
                total = total- np.mean(total)
                total = total / np.max(total)
                '''
                
        return total
    
    
    def plot_on_off_filter(on_center_filter, off_center_filter):
        plt.figure()
        plt.colorbar(plt.pcolor(on_center_filter))
    
    
        plt.figure()
        plt.colorbar(plt.pcolor(off_center_filter))
        
        
    #Multiply weights and images
    def Weights_images(img,weight):
        result = np.dot(img,weight)
        return result
        
        
    
    
    
    
    
    #6 - Implement the Spiking signal in pg 11 of 44 formula and make the raster plot of it against pixels (figure 12)
    #Keep the bins spaced at 10bins (10msecond) (gap = 2millisecond)
    
    def spike_train(img):
        G = params.GAMMA 
        
        pixel_H, pixel_W = img.shape
        delay = np.zeros((img.shape))
        spike_img = np.zeros((img.shape))
        
        for i in range(pixel_H):
            for j in range(pixel_W):
                if img[i][j] > G:
                    spike_img[i][j] = img[i][j]
                    delay[i][j] = 1 / img[i][j]
        #plt.figure()
        #plt.imshow(spike_img)
        
        #return delay
        return spike_img
                    
    
    
    
    
    
    #7 - Raster Plot 1 - Neurons against time (figure 3)
    def raster_one(img):
        neuralData = img
        
        # Draw a spike raster plot
        plt.figure()
        plt.eventplot(neuralData)
        
        # Provide the title for the spike raster plot
        plt.title('Spike raster plot')
        
        #x axis fo the spike raster plot
        plt.xlabel('Neuron')
        
        #Y axis label for the spike raster plot
        plt.ylabel('Spike')
        
        #display the spike raster plot
        plt.show()
    
    
    
    #8 - Raster Plot 2 - Pixel per neuron against time (Figure 12)
    def raster_two(img):
        
        G = params.GAMMA 
        
        pixel_H, pixel_W = img.shape
        delay = np.zeros((img.shape))
        
        for i in range(pixel_H):
            for j in range(pixel_W):
                if img[i][j] > G:
                    
                    delay[i][j] = 1 / img[i][j]
        
        return delay
                    
        
        
    #9 - Convolution layers and STDP 
    def new_plot_DOG(sigma1, sigma2):
       #Fix scale - Problem 
       
        dim = 5
        
        ctotal = np.zeros((dim,dim))
        
        for i in range(dim):   #Leo - 5 x 5 filter size comes from page 5 of 44
            for j in range(dim):
                
                frac1 = 1/(2 * np.pi * sigma1**2)
                frac2 = 1/(2 * np.pi * sigma2**2)
                
                expp = ((i)**2)  + ((j)**2)
                
                #if((-3<=i and i>=3) and (-3<=j and j>=3) )
                ctotal[i][j] = (frac1 * np.exp(-expp/(2*sigma1**2))) - (frac2 * np.exp(-expp/(2*sigma2**2)))
                
                '''
                #Added lines
                total = total- np.mean(total)
                total = total / np.max(total)
                '''
                
        return ctotal
  
        
    
    def convolution_3x3(img, kernel):
        (dim_x, dim_y) = img.shape
        (ker_x, ker_y) = kernel.shape
        gam = 15

        matriz_convolucionada = np.zeros((dim_x, dim_y))

        for i in range(dim_x):
            for j in range(dim_y):
                resultado = 0
                for x in range(0, 4):
                    try:
                        if i + x not in range(dim_x):
                            raise ValueError()
                        for y in range(0, 4):
                            try:
                                if j + y not in range(dim_y):
                                    raise ValueError()

                                resultado += img[i + x, j + y] * kernel[x + 1][y + 1]
                                '''
                                Para el kernel sumo un 1 a cada índice para que lo corra desde 0 hasta 2 y no de -1 a 1
                                For the kernel I add a 1 to each index so that it runs from 0 to 2 and not from -1 to 1
                                '''
                            except ValueError:
                                pass
                    except ValueError:
                        pass
                if img[i][j] > gam:
                    matriz_convolucionada[i][j] = resultado
        return matriz_convolucionada
        
    
    
    
    
    #10 Implement Reward and Punishment (Page 7 of 44)
    
    
    
    
    #11 Implement Learning
    
    
    
    
    
    
    


if __name__ == "__main__":
    
    start = T.time()*1000 #Global time - time() function returns time in seconds
    
    print ("Start time:", start)
    
    img = imageio.imread("2.png")
    print(img)
    
    SNN.weights()
    d=SNN.weights()
    
   
    on_center_filter = SNN.plot_DOG(1,2)
    off_center_filter = SNN.plot_DOG(2,1)
    
    SNN.plot_on_off_filter(on_center_filter, off_center_filter)
    
    
    #Convolve with the on and off filter
    #Add Cmake grey for grey image - plt.imshow(img, cmap='gray')
    plt.figure()
    plt.imshow((SNN.convolution(img, on_center_filter)),cmap='gray')
    
    plt.figure()
    plt.imshow((SNN.convolution(img, off_center_filter)),cmap='gray')
    
    '''
    comb = SNN.Weights_images(img,d)
    #print(comb)
    plt.imshow(comb)
    '''
    
    
    #Spiking image
    plt.figure()
    plt.imshow(SNN.spike_train(img),cmap='gray')
    
    
    
    #Raster one - figure 3
    SNN.raster_one(img)
    
    
    
    #RAster 2 - figure 12
    SNN.raster_two(img)

    spi = img
    plt.figure()
    plt.eventplot(spi)
    
    # Provide the title for the spike raster plot
    plt.title('Spike raster plot')
        
    #x axis fo the spike raster plot
    plt.xlabel('Time')
        
    #Y axis label for the spike raster plot
    plt.ylabel('Pixel/Neuron')
    
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
    
    cImg = SNN.convolution_3x3(SNN.spike_train(img), on_center_filter)
    
    plt.figure()
    plt.imshow(cImg)
    
    
    
    print ("total processing time in milliseconds: ", ((T.time()*1000) - start))
    