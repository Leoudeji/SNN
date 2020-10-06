# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 01:19:04 2020

@author: ludej
"""
#Source: https://www.youtube.com/watch?v=_1BCSr2u05o
def main():
    num = 0
    #input_file = open('weights.txt', 'r')
    #input_file = open('weights_training.txt', 'r')
    input_file = open('train6.txt', 'r')
    
    record = input_file.readline()
    #record = record.rstrip('\n')
    
    print(type(record))
    #print(record.read())
    
    #record = int(record)
    #print(type(record))
    print(record)
    
    """
    while record != "":
        record = int(record)
        num += 1
        
        #record = input_file.readline()
        #record = record.rstrip('\n')
        
        print(num)
        
        input_file.close()
        
    """
        
    
    
    """
    a = 0
    for i in record: 
          
        # Checking for the digit in  
        # the string 
        #if num.isdigit() == True: 
        if i.isnumeric() == True: 
              
            #a += int(i) 
            a += 1
  
    print("The sum is:", a) 
    
    """
        
main()
    