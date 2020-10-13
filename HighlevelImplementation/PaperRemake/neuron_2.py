# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:44:06 2020

@author: ludej
"""

#In this file we define functions used in our program

from params_1 import params as pm
import matplotlib.pylab as plt
import numpy as np

class Neuron:
  def __init__(self, weights=0):
    np.seterr(all='ignore')
    self.input = 0
    self.value = 0
    self.output = 0
    self.threshold = 0
    self.fired = False
    self.potential = 0
    self.weights = np.array([self.init_weight(weights) for x in range(weights)], dtype='float64')

  def fire(self):
    self.fired = True if (self.value > self.threshold) else False
    if self.fired:
      self.value = 0
    return 1 if self.fired else 0

  def init_weight(self, num_weights):
    mu, sigma = 0.5, 0.05 # mean and standard deviation - values come from page 4 of 44
    s = np.random.normal(mu, sigma, 1000)
    return s

  def solve(self):
    raise NotImplementedError("A neuron model needs a solve method")