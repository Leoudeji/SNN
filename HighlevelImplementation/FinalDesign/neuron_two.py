# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:04:35 2020

@author: User

This is the first layer of the Spiking Neural Network. The neuron class takes
in input and makes use of initialised parameters and methods to detect spikes
and appy lateral inhibition when needed.

"""

import random
import numpy as np
from matplotlib import pyplot as plt
from standardVal_one import standardVal as par

class neuron:
    def __init__(self):
        self.t_ref = 30
        self.t_rest = -1
        self.p = par.Prest
        self.Prest = par.Prest
        
    def check(self):
        if self.p >= self.Pth:
            self.p = self.Prest
            return 1
        elif self.p < par.Pmin:
            self.p = par.Prest
            return 0
        else:
            return 0
        
    def inhibit (self):
        self.P = par.Pmin
        
    def initial(self, th):
        self.Pth = th
        self.t_rest = -1
        self.P = par.Prest
    

