# Imports here

from __future__ import print_function, division
import json
import sys
import ImageClassifierFunctions as icfs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import json
import ImageClassifierFunctions as icfs

data_dir = './'
train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

def trainer(pretrained_model="VGG16", epochs=10, gpu= False, learning_rate=0.01):

	data_dir = './'
	train_dir = 'train'
	valid_dir = 'valid'
	test_dir = 'test'
	crit = icfs.nn.CrossEntropyLoss()
	epochs = epochs
	gpu = gpu

	with open('cat_to_name.json', 'r') as f:
	    cat_to_name = json.load(f)

	if pretrained_model == "VGG16":
		algo = models.vgg16(pretrained=True)
	elif pretrained_model == "VGG19":
		algo = models.vgg19(pretrained=True)
	
	#Freezing all layers as the feedback
	for params in algo.parameters():
		params.requires_grad = False
	
	classifier = nn.Sequential(OrderedDict([
                          ('0', nn.Linear(25088, 4096)),
     #About the dimensions of my hidden layers
    #the fully connected layers in the pretrained models have the starting layer dimension =25088 and 4096
    #Since the output from the non-fc layers is 25088, I must use this and define my fc layer to be trained
    #otherwise, a shape mismatch error will occur
                          ('1', nn.ReLU()),
                          ('2', nn.Dropout(0.5)),
                          ('3', nn.Linear(4096, 1000)),
                          ('4', nn.ReLU()),
                          ('5', nn.Dropout(0.5)),
                          ('6', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
	algo.classifier = classifier
	
	opto = optim.SGD(algo.classifier.parameters(), lr=learning_rate, momentum = 0.9)
	learning_rate_scheduler = lr_scheduler.StepLR(opto, step_size=7, gamma=0.1)

	print("here")
	trained_model = icfs.train_model(model=algo, criterion=crit, optimizer=opto,scheduler= learning_rate_scheduler, epochs = epochs, gpu=gpu)
	#save model

	#torch.save(trained_model, "trained_model.pt")
	print("Training done and model saved with name trained_model.pt!")

	ckpoint = {'classifier_input_size': 25088,
              'output_size': len(cat_to_name),
              'optimizer_state' : opto.state_dict(),
              'epochs' : epochs,
              'classifier' : classifier,
              'state_dict': trained_model.state_dict(),
             'data_transforms' : icfs.data_transforms['test'],
             'class_to_idx' : icfs.image_datasets['train'].class_to_idx,
             'model_name': pretrained_model
             }

	torch.save(ckpoint, 'checkpoint.pth')
	print("Training done and checkpoint saved with name checkpoint.pth")





if __name__ == "__main__":
	try:
		pretrained_model = (sys.argv[1])
		epochs = int(sys.argv[2])
		gpu = (sys.argv[3])
		learning_rate = float(sys.argv[4])
		trainer(pretrained_model=pretrained_model, epochs=epochs, gpu =gpu, learning_rate = learning_rate)
	except IndexError:
		print("Rerun the command giving the parameters in the following fashion:\n")
		print("python3 train.py pretrained_model epochs gpu learning_rate")
		print("pretrained_model : choose from 'VGG16' or 'VGG19'\n epochs : choose any number above 0 \n gpu : True or False\n learning_rate: any value between 0 to 1")
