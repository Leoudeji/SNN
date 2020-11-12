
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
import pdb




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
                
                
                #Added lines
        total = total - np.mean(total)
        total = total / np.max(np.abs(total))
                
                
        return total
    
    
    def plot_on_off_filter(on_center_filter, off_center_filter):
        plt.figure()
        plt.colorbar(plt.pcolormesh(on_center_filter))
    
    
        plt.figure()
        plt.colorbar(plt.pcolormesh(off_center_filter))
        
        
    #Multiply weights and images
    def Weights_images(img,weight):
        result = np.dot(img,weight)
        return result
        
        
    
    
    
    
    
    #6 - Implement the Spiking signal in pg 11 of 44 formula and make the raster plot of it against pixels (figure 12)
    #Keep the bins spaced at 10bins (10msecond) (gap = 2millisecond)
    
    def spike_img(img):
        G = params.GAMMA
        pixel_H, pixel_W = img.shape
        spike_img = np.zeros((img.shape))
        
        for i in range(pixel_H):
            for j in range(pixel_W):
                if ((img[i][j])) > G:
                    #spike_img[i][j] = img[i][j]
                    spike_img[i][j] = 1
                  
        
        return spike_img
    
    
    
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
    
        
    def plot_spike_train(img_list):
        plt.figure()
        for i in range(len(img_list)):
            img = np.where(img_list[i].flatten()==1)[0]
            try:
                if len(img) > 0:
                    plt.plot([i]*len(img), img, 'p', c='b')
            except:
                print(img)
    
    
    
    
    
    
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
    
    
                  
        
        
    #9 - New Kernel
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
  
        
    '''
    def convolution_3x3(img, ctotal):
        (dim_x, dim_y) = img.shape
        (ker_x, ker_y) = ctotal.shape
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

                                resultado += img[i + x, j + y] * ctotal[x + 1][y + 1]
                                
                                #Para el kernel sumo un 1 a cada índice para que lo corra desde 0 hasta 2 y no de -1 a 1
                                #For the kernel I add a 1 to each index so that it runs from 0 to 2 and not from -1 to 1
                                
                            except ValueError:
                                pass
                    except ValueError:
                        pass
                if img[i][j] > gam:
                    matriz_convolucionada[i][j] = resultado
        return matriz_convolucionada
    '''
    
    
    
    def convolution_3x3(spi_img, ctotal):
        '''
        if len(spi_img.shape)==2: #1 channel
            spi_img = np.expand_dims(spi_img, axis=2)
        '''
        
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        
        npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #padded image
        #cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                    
                        #Run filter across image
                        for m in range(fc):
                            for k in range(fh):
                                for l in range(fw):
                                    if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                                        #pdb.set_trace()
                                        #npad_img[i][j][n] += spi_img[i + k - fh][j+l-fw][n] * ctotal[k][l][n] #changed npad_img[i][j][m] to npad_img[i][j][n] 
                                        npad_img[i][j][n][Tm] += spi_img[Tm][i + k - fh][j+l-fw][m] * ctotal[k][l][m][n]
                                        
                                        #print(i,j,n,Tm)
                        '''                
                         #Separate this below into a function               
                        if(Tm >= 1):
                            cmpnd_npad_img[i][j][n][Tm] = cmpnd_npad_img[i][j][n][Tm - 1] +  npad_img[i][j][n][Tm]
                            #cmpnd_npad_img[4] = cmpnd_npad_img[Tm - 1] +  npad_img[4]
                        '''
                        
                        '''
                        Test:
                            sum(cImg[1][0,0,0]) should  = cImg[0][0,0,0,-1]
                            cImg[0][1,1,1,-1] should = sum(cImg[1][1,1,1])
                            cImg[1][0,0,0,][0:13].sum() should = cImg[0][0,0,0,12]
                        
                        '''
                                        
                                    
        return npad_img
        #return cmpnd_npad_img, npad_img
        #return spike_img3D
    
    
    
    
    def cumSum(spi_img, ctotal, npad_img):
        
        
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                    
                        #Separate this below into a function               
                        if(Tm >= 1):
                            cmpnd_npad_img[i][j][n][Tm] = cmpnd_npad_img[i][j][n][Tm - 1] +  npad_img[i][j][n][Tm]
                        
                        
                        
                        '''
                        Test:
                            sum(cImg[1][0,0,0]) should  = cImg[0][0,0,0,-1]
                            cImg[0][1,1,1,-1] should = sum(cImg[1][1,1,1])
                            cImg[1][0,0,0,][0:13].sum() should = cImg[0][0,0,0,12]
                        
                        '''
        
        
        return cmpnd_npad_img
    
    
    
    
    def threeDImage(cmpnd_npad_img):
        #I did two things in one shot here
        #First, this was designed to append the image into a 2D image, 
        #then we make a spike image from the appended image
        
        gam = 15
        
        inputImg = cmpnd_npad_img
        
        img_pixel1, img_pixel2, n_layers,t = np.shape(cmpnd_npad_img)
        
        
        #image = np.zeros((img_pixel1, img_pixel2)) 
        spike_3Dimg = np.zeros((img_pixel1, img_pixel2)) 
           
         
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                        
                        
                        if ((inputImg[i][j][n][Tm])) > gam:
                        
                            #image[i][j] += inputImg[i][j][n][Tm]
                            spike_3Dimg[i][j] += 1
            #print(Tm)
             
            #To find the max spike value use: max(fImage.flatten())            
                        
                        
    
        return spike_3Dimg
    
    
    
   
    
    
    #Cov 3x3 using external libraries
    def OptConv_3x3(spi_img, ctotal):
        
        convImg = np.convolve(spi_img, ctotal)
        
        
        return convImg
    
    
    def Linhibit(spi_img, ctotal):
        
        gam = 15 
        
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        
        npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #padded image
        cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        spike_img3D = np.zeros((img_pixel1, img_pixel2, n_layers))
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                    
                        #Run filter across image
                        for m in range(fc):
                            for k in range(fh):
                                for l in range(fw):
                                    if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                                        #pdb.set_trace()
                                        #npad_img[i][j][n] += spi_img[i + k - fh][j+l-fw][n] * ctotal[k][l][n] #changed npad_img[i][j][m] to npad_img[i][j][n] 
                                        npad_img[i][j][n][Tm] += spi_img[Tm][i + k - fh][j+l-fw][m] * ctotal[k][l][m][n]
                                        
                                        if(Tm >= 1):
                                            cmpnd_npad_img[i][j][n][Tm] = npad_img[i][j][n][Tm - 1] + npad_img[i][j][n][Tm]
                                        
                                        #3D spike image
                                        if ((cmpnd_npad_img[i][j][n][Tm])) > gam:
                                            spike_img3D[i][j][n] = 1
                                            
        return spike_img3D[i][j][n]
    
    
    
    
  
    
    
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
    
    #Plot the final iamge
    plt.figure()
    plt.imshow(fImage)
    
    
    '''
    plt.figure()
    #plt.imshow(cImg[:,:,:])  #Will need to work on this
    #plt.colorbar()
    #plt.plot()
    plt.show(cImg)
    '''
    
   
    
    
    
    
    print ("total processing time in milliseconds: ", ((T.time()*1000) - start))
    