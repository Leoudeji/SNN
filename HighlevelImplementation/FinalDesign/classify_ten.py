# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:43:52 2020

@author: ludej

This algorithm detects the content of our picture input
This is another script which regulates the behavior of the neurons
"""

import imageio
import numpy as np
from neuron_two import neuron
from fixedVal_one import fixedVal as par
from receptive_field_four import rf
from spike_train_five import encode2
from synapse_three import learned_weights
#from synapse_three import learned_weights_x, learned_weights_o, learned_weights_synapse
import random

#Parameters
global time, T, dt, t_start, t_end, w_min
time = np.arange(1, par.T+1, 1)

layer2 = []

#create the hidden layer of neurons
for i in range(par.n):
    a = neuron()
    layer2.append(a)
    
#matrix of synapse
synapse = np.zeros((par.n,par.m))


#learned weights
weight_matrix = learned_weights()
for i in range (par.num_of_images):
	synapse[i] = weight_matrix[i]

#random initialization for rest of the synapses
for i in range(par.num_of_images,par.n):
	for j in range(par.m):
		synapse[i][j] = random.uniform(par.w_min,par.w_max)

for k in range(par.epoch):
    #for i in range(1,7):
	for i in range(0,10):
		spike_count = np.zeros((par.n,1))

		#read the image to be classified
		img = imageio.imread("oldTesttrain/test/" + str(i) + ".png")
        
        #img = imageio.imread("newTestTrain/test/" + str(i) + ".png")

    	#initialize the potentials of output neurons
		for x in layer2:
			x.initial()

    #calculate teh membrane potentials of input neurons
		pot = rf(img)

    #generate spike trains. Select between deterministic and stochastic
		# train = np.array(encode_deterministic(pot))
		train = np.array(encode2(img))

    #flag for lateral inhibition
		f_spike = 0
		active_pot = np.zeros((par.n,1))
		for t in time:
			for j, x in enumerate(layer2):
				active = []

        #update potential if not in refractory period
				if(x.t_rest<t):
					x.P = x.P + np.dot(synapse[j], train[:,t])
					if(x.P>x.Prest):
						x.P -= x.D
					active_pot[j] = x.P

			# Lateral Inhibition
			if(f_spike==0):
				high_pot = max(active_pot)
				if(high_pot>par.Pth):
					f_spike = 1
					winner = np.argmax(active_pot)
					for s in range(par.n):
						if(s!=winner):
							layer2[s].P = layer2[s].Pmin

			#Check for spikes
			for j,x in enumerate(layer2):
				s = x.check()
				if(s==1):
					spike_count[j] += 1
					x.t_rest = t + x.t_ref
		print (spike_count)



'''
#learned weights
synapse[0] = learned_weights_x()
synapse[1] = learned_weights_o()

#random initialization for rest of the synapses
for i in range(par.n):
    synapse[i] = learned_weights_synapse(i)
    
for k in range(1):
    for i in range(3):
        spike_count = [0,0,0,0]
        
        #read the image to be classified
        img = imageio.imread("test/{}.png".format(i))
        
        #initialize the potentials of output neurons
        for x in layer2:
            x.initial(par.Pth)
            
        #calculate the membrane potentials of the input neurons
        pot = rf(img)
        
        #generate sike trains
        train = np.array(encode(pot))
        
        #flag for lateral inhibition
        f_spike = 0
        
        active_pot = [0,0,0,0]
        
        for t in time:
            for j, x in enumerate(layer2):
                active = []
                
        #update potential if not in refractory period
                if(x.t_rest<t):
                    x.P = x.P + np.dot(synapse[j], train[:,t])
                    if(x.P>par.Prest):
                        x.P -= par.D
                    active_pot[j] = x.P
                    
             
            #Lateral inhibition
            if(f_spike==0):
                high_pot = max(active_pot)
                if(high_pot>par.Pth):
                    f_spike = 1
                    winner = np.argmax(active_pot)
                    print(i, winner)
                    for s in range(par.n):
                        if(s!=winner):
                            layer2[s].P = par.Prest
                            
                            
            #check for spikes
            for j,x in enumerate(layer2):
                s = x.check()
                if(s==1):
                    print(j,s)
                    spike_count[j] += 1
                    x.t_rest = t + x.t_ref
        print(spike_count)
        
'''