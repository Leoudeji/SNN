# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:49:56 2020
source: https://www.geeksforgeeks.org/extract-numbers-from-a-text-file-and-add-them-using-python/

@author: ludej
"""

# Python program for reading 
# from file 
  

#h = open('weights.txt', 'r') #41123
#h = open('weights_training.txt', 'r') #576107
#h = open('train6.txt', 'r') #ans = 505
  
# Reading from the file 
content = h.readlines() 
  
# Varaible for storing the sum 
a = 0
  
# Iterating through the content 
# Of the file 
for line in content: 
      
    for i in line: 
          
        # Checking for the digit in  
        # the string 
        if i.isdigit() == True: 
              
            a += int(i) 
  
print("The sum is:", a) 




"""
k = open("weights.txt",'r') 
#k = open("weights_training.txt",'r') 
#k = open("train6.txt",'r') 

lines = k.readlines()     #i am reading lines here 
counter = 0          #counter update each time number is entered 
for line in lines:            #taking each line 
   conv_int = int(line)         #converting string to int 
   counter = counter + conv_int      #update counter 
print(counter) 

"""

"""
allNums = [] 
total = 0   #/Users/ludej/OneDrive/Desktop/Spring2020/Research/SummerWork/Week12_August24/HighlevelImplementation/SNN/HighlevelImplementation/FinalDesign
with open(r"weights.txt", "r+") as f: 
    data = f.readlines() # read the text file 
    for line in data: 
        allNums += line.strip().split(" ") # get a list containing all the numbers in the file 
    print (allNums) # printing list to see if all numbers exist, Kindly comment if not needed coz with huge data the any ide may hang 
    for num in allNums: 
        total += int(num) 
    print (total) # printing the final sum 
    
    """