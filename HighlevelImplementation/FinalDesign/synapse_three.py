# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:32:13 2020

@author: ludej
This script is used to learn weights
"""

#Read and return weights produced by spike_train_five.py for  matching synapses
def learned_weights_x():
    ans = []
    with open('weights.txt', 'r') as weight_file:
        lines = weight_file.readlines()
        for i in lines[0].split('\t'):
            ans.append(float(i))
    return ans
    
#Read and return weights produced by spike_train_five.py for matching synapses
def learned_weights_o():
    ans = []
    
    with open('weights.txt', 'r') as weight_file:
        lines = weight_file.readlines()
        for i in lines[1].split('\t'):
            ans.append(float(i))
    return ans 
    
def learned_weights_synapse(id):
    ans = []
    with open('weights.txt', 'r') as weight_file:
        lines = weight_file.readlines()
        if (len(lines) <= id):
            return ans
        for i in lines[id].split('\t'):
            ans.append(float(i))
    return ans
    
    
#Show that we read the wieghts and processed them into a sequence of feed to be used for classification
if __name__ == '__main__':
    a = learned_weights_x()
    print(a)
    
    #I could add the below:
    #b = learned_weights_o()
    #c = learned_weights_synapse(id)
    