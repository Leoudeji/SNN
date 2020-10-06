# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:32:13 2020

@author: ludej
This script is used to append initial trained weight values to our network. 
It also contains the names of images that will be used in this file
"""

#Read and return weights produced by spike_train_five.py for  matching synapses

import imageio

#Reads the returns the weights produced by training
#could use weights from weights.txt file
def learned_weights():
	image_names = ["0","1", "2", "3", "4", "5", "6","7","8","9"]
	ans = []
	for image in image_names:
		temp = []
		img = imageio.imread("oldTesttrain/training/" + image + ".png")
		for i in img:
			for j in i:
				if(j==0):
					temp.append(-0.7)
				else:
					temp.append(1)
		ans.append(temp)
	return ans
    
    
#Show that we read the wieghts and processed them into a sequence of feed to be used for classification
if __name__ == '__main__':
    #a = learned_weights_x()
    a = learned_weights()
    print(a)
    
    #I could add the below:
    #b = learned_weights_o()
    #c = learned_weights_synapse(id)
    