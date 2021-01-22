# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:42:42 2020

@author: User
"""
#1: https://numpy.org/devdocs/user/absolute_beginners.html
#2: https://numpy.org/doc/stable/reference/arrays.ndarray.html
    
from scipy import signal
import numpy as np
#test

s = np.arange(5);  
w = [1,2,3]

s2 = np.array([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
w2 = np.array([[1, 2], [5, 6]])

s3 = np.arange(18).reshape((2,3,3))
w3 = np.arange(8).reshape((2,2,2))



def sci_con1():
    res = signal.convolve(s,w[::-1], mode = 'valid')
    
    return res


def sci_con2():
    res = signal.convolve(s2,w2[::-1,::-1], mode = 'valid')
    
    return res

def sci_con3():
    res = signal.convolve(s3,w3[::-1,::-1,::-1], mode = 'valid')
    
    return res


def testCon1(s,w):
    
    #res  = np.zeros(len(s)-len(w)+1)
    
    #Ls= s.shape
    #write convolution
    
    #Make valid mode
    
    
    
    img_pixel1 = s.shape[0] #get dimension of input image
    fh= np.shape(w)[0] #get dimension of 2d filter
        
    imp = img_pixel1-fh+1
    
    pad_img = np.zeros((imp)) #padded image
        
    for i in range(imp): #Loop over image width
      
                
            #Run filter across image
            for k in range(fh): #Loop over filter width
               
                        
                    #The 2 lines below perform convolution
                    #if 0 <= (i+k)-fh <= imp:  #previous
                    if (i+k) >= 0 and (i+k) <= img_pixel1:
                        pad_img[i] += s[i + k] * w[k]
                        
                        '''
                        when i = 0
                        s[0 -3] * w[0]
                        s[1 -3] * w[1]
                        s[2 -3] * w[2]
                        
                        '''
    
    
    return pad_img


def testCon2(s2,w2):
    
    #res  = np.zeros(len(s)-len(w)+1)
    
    #Ls, ws = s2.shape
    #write convolution
    
    img_pixel1, img_pixel2 = s2.shape #get dimension of input image
    fh, fw = w2.shape #get dimension of 2d filter
    
    imp = img_pixel1-fh+1
        
    #pad_img = np.zeros((img_pixel1, img_pixel2)) #padded image
    pad_img = np.zeros((imp, imp)) #padded image
        
    #for i in range(img_pixel1): #Loop over image width
        #for j in range(img_pixel2): #Loop over image height
    for i in range(imp): #Loop over image width
        for j in range(imp): #Loop over image height
                
            #Run filter across image
            for k in range(fh): #Loop over filter width
                for l in range(fw): #Loop over filter height
                        
                    #The 2 lines below perform convolution
                    #if 0 <= ((i+k)-fh < img_pixel1) and 0 <= ((j+l)-fw < img_pixel2): #previous
                    if (i+k) >= 0 and (i+k) <= img_pixel1 and (j+l) >= 0 and (j+l) <= img_pixel2:
                        #pad_img[i][j] += s2[i + k - fh][j+l-fw] * w2[k][l]
                        pad_img[i][j] += s2[i + k ][j+l] * w2[k][l]
    
    
    #np.abs(x-y).sum()
    #Pass-in the same input an check result
    
    return pad_img


def testCon3(s3,w3):

    chanls, img_pixel1, img_pixel2 = s3.shape
    #t,img_pixel1, img_pixel2,chanls =  np.shape(spi_img)  #spi_img.shape (I used np.array to covert it to an array)
    #fh, fw, fc, n_layers= ctotal.shape
    fc, fh, fw = w3.shape
        
    assert(chanls == fc) #had to update the dimension of my img to match that of the filter. Then update the use of "n_layers" below
    
    hgh = img_pixel1-fh+1
    wdth = img_pixel2-fw+1
        
    #npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #padded image
    #npad_img = np.zeros((hgh, wdth, n_layers,t)) #padded image  #Previous
    #cmpnd_npad_img = np.zeros((img_pixel1, img_pixel2, n_layers,t)) #cumulative padded  image
        
    #npad_img = np.zeros((hgh, wdth, fc))   
    npad_img = np.zeros((hgh, wdth)) 
    
    '''
    for Tm in range(t):
        for n in range(n_layers): #Loop over all layers
                
                #for i in range(img_pixel1):
                    #for j in range(img_pixel2):
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
                            
                            #npad_img[i][j][m] += s3[m][i + k][j+l] * w3[m][k][l]   
                            npad_img[i][j] += s3[m][i + k][j+l] * w3[m][k][l]   
    
     
     
    return npad_img



def testLinhibit():
    #gam = 15
    
    a = np.random.uniform(size=(2,3,4,5))
    img_pixel1, img_pixel2, n_layers,t = a.shape
    maxim = np.zeros((30))
    
             
    for i in range(img_pixel1):
        for j in range(img_pixel2):
                    
            for Tm in range(t):
                for n in range(n_layers): #Loop over all layers
            
                    #findMax = max(inputImg[i,j,:,t-1])
                    findMax = max(a[i,j,:,Tm])
                    maxim = a[i,j,:,Tm] #np.size(maxim) returns 4 here becuase the size of our answer is 4 (i.e. 4 maps)
                    
                    '''
                    printing maxim = a[i,j,:,Tm] gave:
                        array([0.92491506, 0.75233455, 0.30016982, 0.35181613])
                    '''
                    
    return findMax
                    
                    