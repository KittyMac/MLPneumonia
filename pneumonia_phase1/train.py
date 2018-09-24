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
from keras.callbacks import ModelCheckpoint
import random
import time
import sys
import math

import signal
import time
import coremltools

from shutil import copyfile

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
			trainingFiles.append( ("train/"+file, True) )
	
	for file in os.listdir("not_train/"):
		if file.endswith(".npy"):
			trainingFiles.append( ("not_train/"+file, False) )
	
	num = len(trainingFiles)
	input = np.zeros((num,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]), dtype='float32')
	output = np.zeros((num,1), dtype='float32')
	
	checkpoint = ModelCheckpoint(model.MODEL_H5_NAME, monitor='loss', verbose=1, save_best_only=True, mode='min')
	
	for n in range(0,100):
		random.shuffle(trainingFiles)
	
		for i in range(0,num):
			trainingFile = trainingFiles[i]
			np.copyto(input[i], np.load(trainingFile[0]))
			if trainingFile[1] == True:
				output[i][0] = 1
			else:
				output[i][0] = 0
	
		_model.fit(input,output,batch_size=32,shuffle=True,epochs=4,verbose=1,callbacks=[checkpoint])
		
		copyfile(model.MODEL_H5_NAME, "../pneumonia_phase2/%s" % (model.MODEL_H5_NAME))
			

if __name__ == '__main__':
	Learn()
	