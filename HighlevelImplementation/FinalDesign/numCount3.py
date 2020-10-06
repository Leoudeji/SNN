# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 02:04:50 2020

@author: ludej
"""

# Python program for reading 
# from file 
 # Source: https://www.geeksforgeeks.org/extract-numbers-from-a-text-file-and-add-them-using-python/
  
#h = open('weights.txt', 'r')
#h = open('weights_training.txt', 'r') 
h = open('train6.txt', 'r')

#print(type(h))

  
# Reading from the file 
content = h.readlines() 
print(type(content))
print(content)
  
# Varaible for storing the sum 
a = 0
  
# Iterating through the content 
# Of the file 
for line in content: 
      
    for i in line: 
          
        # Checking for the digit in  
        # the string 
        if i.isdigit() == True: 
        #if i.isnumeric() == True: 
              
            a += int(i) 
  
print("The sum is:", a) 