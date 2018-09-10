from __future__ import division

import os

# Note: we're trying out PlaidML
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import backend as keras

from keras.preprocessing import sequence
import numpy as np
import coremltools
import model
import data
import json
import operator
import keras.callbacks
import random
import time
import sys
import math

import signal
import time
import coremltools

from model import IMG_SIZE

from PIL import Image,ImageDraw

def Learn():
		
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. load all of the samples (if we can...)
	print("loading samples")
	trainingFiles = []
	for file in os.listdir("train/"):
		if file.endswith(".npy"):
			trainingFiles.append( (file, file.startswith("1.")) )
	
	num = len(trainingFiles)
	input = np.zeros((num,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]), dtype='float32')
	output = np.zeros((num,2), dtype='float32')
	
	class_weight = {
		0:0,
		1:0
	}
	for trainingFile in trainingFiles:
		if trainingFile[1] == True:
			class_weight[0] += 1
		else:
			class_weight[1] += 1
	
	class_weight[0] /= len(trainingFiles)
	class_weight[1] /= len(trainingFiles)
	
	print("class_weight", class_weight)
	
	for n in range(0,100):
		random.shuffle(trainingFiles)
	
		for i in range(0,num):
			trainingFile = trainingFiles[i]
			np.copyto(input[i], np.load("train/%s" % (trainingFile[0])))
			if trainingFile[1] == True:
				output[i][0] = 0
				output[i][1] = 1
			else:
				output[i][0] = 1
				output[i][1] = 0
	
		_model.fit(input,output,batch_size=128,shuffle=True,epochs=2,class_weight=class_weight,verbose=1)
	
		_model.save(model.MODEL_H5_NAME)
		_model.save("../%s" % (model.MODEL_H5_NAME))
	

if __name__ == '__main__':
	Learn()
	