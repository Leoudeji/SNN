# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:24:05 2021

@author: User
"""

import numpy as np
from scipy import signal #Used to verify 3d convolution (test_Conv_3x3)


    
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