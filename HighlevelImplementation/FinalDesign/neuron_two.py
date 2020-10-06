# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:04:35 2020

@author: User

This is the first layer of the Spiking Neural Network. The neuron class takes
in input and makes use of initialised parameters and methods to detect spikes
and apply lateral inhibition when needed.
This class embodies the characteristics and functions of the neuron in our network

"""


from fixedVal_one import fixedVal as par

class neuron:
    def __init__(self):
        self.t_ref = 30
        self.t_rest = -1
        self.P = par.Prest  #P = potential
        self.Prest = par.Prest
        
        #Added today Sept 9
        '''
        self.Pth = par.Pth
        self.D = par.D
        self.Pmin = par.Pmin
        '''
        
    def check(self):
        if self.P >= self.Pth:
            self.P = self.Prest
            return 1
        elif self.P < par.Pmin:
            self.P = par.Prest
            return 0
        else:
            return 0
        
    def inhibit (self):
        self.P = par.Pmin
        
    #def initial(self, th):
    def initial(self):
        #self.Pth = th
        self.t_rest = -1
        self.P = par.Prest
    
    #Added today sept 29 for learning_nine.py
    def initialize(self,th):
        self.Pth = th
        self.t_rest = -1
        self.P = par.Prest

