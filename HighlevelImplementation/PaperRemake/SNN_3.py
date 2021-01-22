
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:45:19 2020

Link to Study: https://arxiv.org/pdf/1903.12272.pdf
Another study: https://arxiv.org/pdf/1611.01421.pdf
Gitrepo to parent paper: https://github.com/ruthvik92/SpykeFlow/blob/master/spykeflow/network.py

@author: ludej
Advisor: Prof. Martin Margala


"""

#Research Title: FPGA Implementation of Spiking Neural Networks
#Dataset: MNIST daatset


#import necessary libraries
import imageio
import time as T
import matplotlib.pyplot as plt 
import numpy as np
from params_1 import params
#from neuron_2 improt neuron
import pdb # used to debug program
from scipy import signal #Used to verify 3d convolution (test_Conv_3x3)


def f():
    print("Hello2");


#Create a class to contain all the methods in our design
class SNN():  
    
    #1 - Create function to replicate figure 1
    
    
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
    
    
    
    #4 - Convolution (2 dimensional)
    #Perform convolution operation on an input image
    #We use same-mode convolution; so the shape of the output and the input image remains the same (page 5 of 44, Figure 5)
    #How do we incorporate time and thresholding into formula as in page 5?? (spike_img and spike_train methods below does this)
    def convolution(img, total):
        img_pixel1, img_pixel2 = img.shape #get dimension of input image
        fh, fw = total.shape #get dimension of 2d filter
        
        pad_img = np.zeros((img_pixel1, img_pixel2)) #padded image
        
        for i in range(img_pixel1): #Loop over image width
            for j in range(img_pixel2): #Loop over image height
                
                #Run filter across image
                for k in range(fh): #Loop over filter width
                    for l in range(fw): #Loop over filter height
                        
                        #The 2 lines below perform convolution
                        if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                            pad_img[i][j] += img[i + k - fh][j+l-fw] * total[k][l]
                            
        return pad_img

    
    
    
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
    
    
    #Function performs 3D Convolution on stacked spiking image of ON and OFF filter (Pg 12/44)
    #ctotal represents the kernel
    def convolution_3x3(spi_img, ctotal):
        '''
        if len(spi_img.shape)==2: #1 channel
            spi_img = np.expand_dims(spi_img, axis=2)
        '''
        
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        hgh = img_pixel1-fh+1
        wdth = img_pixel2-fw+1
        
        #npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #padded image
        npad_img = np.zeros((hgh, wdth, n_layers,t)) #padded image
        #cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                '''
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                '''
                
                for i in range(hgh):
                    for j in range(wdth):
                    
                        #Run filter across image
                        for m in range(fc):
                            for k in range(fh):
                                for l in range(fw):
                                    #if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2):
                                    if (i+k) >= 0 and (i+k) <= img_pixel1 and (j+l) >= 0 and (j+l) <= img_pixel2:
                                        #pdb.set_trace()
                                        #npad_img[i][j][n] += spi_img[i + k - fh][j+l-fw][n] * ctotal[k][l][n] #changed npad_img[i][j][m] to npad_img[i][j][n] 
                                        
                                        #npad_img[i][j][n][Tm] += spi_img[Tm][i + k - fh][j+l-fw][m] * ctotal[k][l][m][n]  #Previous
                                        
                                        npad_img[i][j][n][Tm] += spi_img[Tm][i + k][j+l][m] * ctotal[k][l][m][n]
                                        
                                        #npad_img[i][j][n][Tm] += spi_img[Tm][(i - k)][(j-l)][m] * ctotal[k][l][m][n]
                                        
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
    
    
    
    
    #Function loops through the result of convolution_3x3() function to 
    #Calculate cumulative sum of image values after 3D convolution
    def cumSum(spi_img, ctotal, npad_img):
        
        
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        hgh = img_pixel1-fh+1
        wdth = img_pixel2-fw+1
        
        cmpnd_npad_img = np.zeros((hgh,wdth, n_layers,t)) #cumulative padded  image
        
        #cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                '''
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                '''
                        
                for i in range(hgh):
                    for j in range(wdth):
                    
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
    
    
    
    
    
    #Converts the results from cumSum() function into a 2D image for plotting (Figure 14 - Page 13/44)
    def threeDImage(cmpnd_npad_img):
        #I did two things in one shot here
        #First, this was designed to append the image into a 2D image, 
        #then we make a spike image from the appended image
        
        gam = 15
        
        inputImg = cmpnd_npad_img
        
        img_pixel1, img_pixel2, n_layers,t = np.shape(cmpnd_npad_img)
        
        
        #image = np.zeros((img_pixel1, img_pixel2)) 
        spike_3Dimg = np.zeros((img_pixel1, img_pixel2)) #add t
        #spike_3Dimg = np.zeros((n_layers, t))
        
        #for every pixel the max should be 30
           
         

        for i in range(img_pixel1):
            for j in range(img_pixel2):
                for Tm in range(t):
                    for n in range(n_layers): #Loop over all layers
                
                        
                        if ((inputImg[i][j][n][Tm])) > gam:
                        
                            #image[i][j] += inputImg[i][j][n][Tm]
                            #spike_3Dimg[i][j] += 1
                            spike_3Dimg[i][j] = inputImg[i][j][n][Tm]
            #print(Tm)
             
            #To find the max spike value use: max(fImage.flatten())            
                        
                        
    
        return spike_3Dimg
    
    
   
    
    
    #Perform 3D Covolution using external libraries
    #This function is used to test the correctness of our original algorithm
    def test_Conv_3x3(spi_img, ctotal):
        
        sa,sb,sc,sd = np.transpose(spi_img,axes=[1,2,3,0]).shape
        ca,cb,cc,cd = np.transpose(ctotal,axes=[0,1,2,3]).shape
        
        #nspi_img = np.asarray(np.transpose(spi_img,axes=[1,2,3,0]))
        #nctotal = np.asarray(np.transpose(ctotal,axes=[0,1,2,3]))
        nspi_img = np.transpose(spi_img,axes=[1,2,3,0])
        nctotal = np.transpose(ctotal,axes=[0,1,2,3])
        convImg = np.zeros((24,24,30,20))
        
        #nspi_img = np.transpose(spi_img,axes=[0,3,1,2])
        #nctotal = np.transpose(ctotal,axes=[3,2,0,1]) 
        
        #convImg = signal.fftconvolve(nspi_img, nctotal, mode='same', axes=[0,1,2])
        #convImg = signal.convolve(nspi_img, nctotal, mode='same', axes=[0,1,2])
        
        
        
        #convImg = signal.convolve(nspi_img, nctotal, mode='same')  #The only one that seems to work ok
        #convImg = signal.convolve(nspi_img, nctotal, mode='valid')
        
        
        for a in range(sd):   #Loop for all time
            for b in range(cd):   #Loop for all layer
                convImg[:,:,b,a] = signal.convolve(nspi_img[:,:,:,a], nctotal[::-1,::-1,::-1,b], mode='valid', method='auto').squeeze()
                #convImg[:,:,cd,sd] = signal.convolve(nspi_img[:,:,:,sd], nctotal[::-1,::-1,::-1,cd], mode='same')
                #convImg = signal.fftconvolve(nspi_img[:,:,:,a], nctotal[::-1,::-1,::-1,b], mode='valid')
                #convImg = signal.convolve(nspi_img, nctotal, mode='valid') 
        
        
        #convImg = signal.fftconvolve(np.transpose(spi_img,axes=[1,2,3,0]), np.transpose(ctotal,axes=[0,1,2,3]), mode='valid')
        #convImg = signal.convolve(np.transpose(spi_img,axes=[1,2,3,0]), np.transpose(ctotal,axes=[0,1,2,3]), mode='full')
        #Valid mode gave an error
        #full mode gives a result with the wrong dimension
        
        #convImg = signal.fftconvolve(np.transpose(spi_img,axes=[1,2,3,0]).shape, np.transpose(ctotal,axes=[0,1,2,3]).shape, mode='valid')
        
        #convImg = signal.fftconvolve(spi_img, ctotal, mode='same')
        #convImg = signal.fftconvolve(spi_img, ctotal, mode='valid', axis =)
        
        #I need to change the alignment (axis) to perform convolution.
        #I might need to use np.transpose
        #Depending on which works better you can use the faster one for the program
        
        
        
        return convImg
    
    
    
    '''
    def tstfunction():
        sp = np.random.uniform(size=(28,28,2,20))
        tp = np.random.uniform(size=(5,5,2,30))
        up = np.zeros(shape=(28,28,30,20))
        
        for a in range(20):
            for b in range(30):
                #up[:,:,a,b] = np.convolve(sp[:,:,:,a], tp[:,:,:,b], mode='valid').squeeze()
                #up[:,:,a,b] = np.convolve(sp[::-1,::-1,::-1,a], tp[::-1,::-1,::-1,b], mode='valid').squeeze()
                up[:,:,a,b] = np.convolve(sp[:,:,:,a], tp[::-1,::-1,::-1,b], mode='valid').squeeze()
        
        
        return up
    '''
    
    
    
    #Converts the results from test_Conv_3x3() function into a 2D image for plotting
    def test3DCSum(convImg):
      

        gam = 15
        
        inputImg = convImg
        
        img_pixel1, img_pixel2, n_layers,t = np.shape(convImg)
        
        
        #image = np.zeros((img_pixel1, img_pixel2)) 
        test_csum  = np.zeros((img_pixel1, img_pixel2)) #add t
        #spike_3Dimg = np.zeros((n_layers, t))
        
        #for every pixel the max should be 30
           
         

        for i in range(img_pixel1):
            for j in range(img_pixel2):
                for Tm in range(t):
                    for n in range(n_layers): #Loop over all layers
                
                        
                        if ((inputImg[i][j][n][Tm])) > gam:
                        
                            #image[i][j] += inputImg[i][j][n][Tm]
                            #spike_3Dimg[i][j] += 1
                           test_csum[i][j] = inputImg[i][j][n][Tm]     
                           
        '''
        t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
        fh, fw, fc, n_layers= ctotal.shape
        
        test_csum = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
        
        assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
        
        
        for Tm in range(t):
            for n in range(n_layers): #Loop over all layers
                
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                    
                        #Separate this below into a function               
                        if(Tm >= 1):
                            test_csum[i][j][n][Tm] = test_csum[i][j][n][Tm - 1] +  convImg[i][j][n][Tm]
        '''
        
        
        return test_csum 
    
    
    
    
    
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
                
                        #findMax = max(inputImg[i,j,:,t-1])
                        findMax = max(inputImg[i,j,:,Tm])
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
    
    
    
    
    #Functons to Implement STDP competition
    
    def STDP_Kernel(u,v):
        
        SK= np.random.normal(0, 0, size=(u,v)) 
        
        
        return SK
    
    
    
    
    #Function creates results in Figure 16 (Page 14/44) - Incomplete
    def STDP_Comp(cmpnd_npad_img):
        
        gam = 15
        
        max_neuron_value = {};
        
        #fh, fw = SK.shape
        inputImg = cmpnd_npad_img
        
        img_pixel1, img_pixel2, n_layers,t = np.shape(cmpnd_npad_img)
            
        stdpCompResult = np.zeros((img_pixel1, img_pixel2)) #padded image
        imgPerMap = np.zeros((img_pixel1, img_pixel2)) #position of maximum potential value
        finalImage = np.zeros((img_pixel1, img_pixel2, n_layers,t))
            
        
        #LIimg3D = np.zeros((img_pixel1, img_pixel2)) #add t
        findMax = np.zeros((img_pixel1, img_pixel2))
        neuron = np.zeros((img_pixel1, img_pixel2, n_layers,t))
        maxValue = 0
        
        
        #for every pixel the max should be 30
         
                
        #The difference from lateral inhibition is 'n' in findMax and position of loop over time
                        
               
        for w in range(n_layers): #Loop over all layers
                   
            for Tm in range(t):
                        
                for i in range(img_pixel1):
                    for j in range(img_pixel2):
                        #n_neuron[str(i)] = 0
                        
                        imgPerMap = inputImg[:,:,w,Tm]
                        
                #imgPerMapList = imgPerMap.reshape(img_pixel1 * img_pixel2,1)
    
                #findMax = max(imgPerMapList)
                    #The result of this "inputImg[:,:,w,:]" is a 24x24x20 cube
            
            
            
            
            for i in range(img_pixel1):
                for j in range(img_pixel2):
                    if ((imgPerMap[i][j]) > gam):
                        findMax[i][j] = imgPerMap[i][j]
                        #maxValue = findMax[i][j].max()
                        maxValue = findMax.max()
                        max_neuron_value[str(w)]= imgPerMap.max()
                        
                        #Question:: max() function is tricky. maxValue = findMax[i][j].max() gave me a diff result. Why??
                        
                    if ((imgPerMap[i][j]) == maxValue):
                        stdpCompResult[i][j]= 1
                        finalImage[i][j][w][Tm] = imgPerMap[i][j]
                        #finalImage[i][j][w][Tm] = 1
                        
                    #else:
                        #stdpCompResult[i][j] = 0
                        
                       
                        
                
                #if ((inputImg[i][j][w][Tm])) > gam:
                
                #neuron = inputImg[i,j,w,Tm]
                #maxPosition = [i, j, w]
                #stdpCompResult[i][j][w] = 1
            
                    
                               
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
                                            
        
        
        return stdpCompResult, findMax, maxValue, max_neuron_value,finalImage
   
    
    
    
    
    
    #Read Multiple Images
    def ReadImages():
        
        for n in range(10000):
            img_Test = imageio.imread("test/" + str(n) + ".png")
            
        for i in range(60000):
            img_Train = imageio.imread("training/" + str(i) + ".png")
        
        return img_Test, img_Train
    
    
    
    
    
    
    
    #10 Implement Reward and Punishment (Page 7 of 44)
    
    
    
    
    #11 Implement Learning
    def learning():
        return 0
    
    
    
    
    #12 Max Pooling
    
    def maxPooliing():
        
        return 0
    
    


if __name__ == "__main__":
#def run():
    
    start = T.time()*1000 #Global time - time() function returns time in seconds
    
    print ("Start time:", start)
    
    img = imageio.imread("2.png")
    #print(img)
    
    SNN.weights()
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
    plt.figure()
    plt.imshow(STDP_CompImg[0])
    plt.figure()
    plt.colorbar(plt.pcolormesh(STDP_CompImg[0]))
    plt.figure()
    plt.imshow(SNN.threeDImage(STDP_CompImg[4]))
    #plt.figure()
    #plt.colorbar(plt.pcolormesh(STDP_CompImg[4]))
    
    
    '''
    plt.figure()
    #plt.imshow(cImg[:,:,:])  #Will need to work on this
    #plt.colorbar()
    #plt.plot()
    plt.show(cImg)
    '''
    
   
    
    
    
    
    print ("total processing time in milliseconds: ", ((T.time()*1000) - start))
    