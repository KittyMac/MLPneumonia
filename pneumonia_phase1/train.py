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

class GracefulKiller:
	kill_now = False
	def __init__(self):
		signal.signal(signal.SIGINT, self.exit_gracefully)
		signal.signal(signal.SIGTERM, self.exit_gracefully)
		signal.signal(signal.SIGUSR1, self.exit_gracefully)

	def exit_gracefully(self,signum, frame):
		self.kill_now = True

def Learn():
	
	killer = GracefulKiller()
		
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. load all of the samples (if we can...)
	print("loading samples")
	trainingFiles = []
	for file in os.listdir("train/"):
		if file.endswith(".npy"):
			trainingFiles.append( ("train/"+file, True) )
	
	for file in os.listdir("not_train/"):
		if file.endswith(".npy"):
			trainingFiles.append( ("not_train/"+file, False) )
	
	num = len(trainingFiles)
	input = np.zeros((num,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]), dtype='float32')
	output = np.zeros((num,1), dtype='float32')
		
	for n in range(0,100):
		random.shuffle(trainingFiles)
	
		for i in range(0,num):
			trainingFile = trainingFiles[i]
			np.copyto(input[i], np.load(trainingFile[0]))
			if trainingFile[1] == True:
				output[i][0] = 1
			else:
				output[i][0] = 0
	
		_model.fit(input,output,batch_size=32,shuffle=True,epochs=4,verbose=1)
	
		_model.save(model.MODEL_H5_NAME)
		_model.save("../pneumonia_phase2/%s" % (model.MODEL_H5_NAME))
		
		if killer.kill_now == True:
			break
	

if __name__ == '__main__':
	Learn()
	