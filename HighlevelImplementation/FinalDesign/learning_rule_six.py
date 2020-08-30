# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:41:05 2020

@author: ludej

Implements Spike Time Dependent Plasticity (STDP) rule for estimating weights and curve
"""
import numpy as np
from matplotlib import pyplot as plt
from fixedVal_one import fixedVal as par

#STDP learning curve reinforcement
def rl(t):
    #This is where the fixed parameters A_plus and tau_plus comes in
    if t>0:
        return -par.A_plus*np.exp(-float(t)/par.tau_plus)
    if t<=0:
        return par.A_minus*np.exp(float(t)/par.tau_minus)



#STDP weight update rule
def update(w, del_w):
    if del_w<0:
        return w + par.sigma*del_w*(w-abs(par.w_min))*par.scale
    elif del_w>0:
        return w + par.sigma*del_w*(par.w_max-w)*par.scale
        
if __name__ == '__main__':
    print(rl(-20)*par.sigma)
        
    
