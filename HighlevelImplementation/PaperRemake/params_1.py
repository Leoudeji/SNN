# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:34:57 2020

@author: ludej
"""

class params:
    #a = 20;
    #b = 30;
    aplus = 0.004 # Value of learning constant. used to update weight (page 3 of 44). These values are supposed to change as learning progresses
    aminus = 0.003
    w = [] #weight value
    t = 0 #time
    
    GAMMA = 50 #Membrane potential (Page 11 of 44)
    