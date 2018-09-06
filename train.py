from __future__ import division

from keras import backend as keras

from keras.preprocessing import sequence
from dateutil import parser
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

from PIL import Image,ImageDraw

######
# allows us to used ctrl-c to end gracefully instead of just dying
######
class SignalHandler:
  stop_processing = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.stop_processing = True
######

def Preprocess():
	# force all of the images to get cached
	generator = data.DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")	
	generator.ignoreCaches = True
	input,output = generator.generateImages(0)

def Learn():
		
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	generator = data.DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")
	
	iterations = 100000
		
	print("beginning training")
	handler = SignalHandler()
	i = 0
	while True:
		
		if handler.stop_processing:
			break
		
		n = 10000
		print(i)
		Train(generator,_model,n)
		i += n
		
		if i >= iterations:
			break
				
	
	_model.save(model.MODEL_H5_NAME)


def Train(generator,_model,n):
	
	train,label = generator.generateImages(n)
	
	batch_size = 32
	if n < batch_size:
		batch_size = n
	
	_model.fit(train,label,batch_size=batch_size,shuffle=True,epochs=1,verbose=1)

def Test():
	_model = model.createModel(True)
	print("test not implemented yet")		
	

if __name__ == '__main__':
	if sys.argv >= 2:
		if sys.argv[1] == "test":
			Test()
		elif sys.argv[1] == "learn":
			Learn()
		elif sys.argv[1] == "preprocess":
			Preprocess()
	else:
		Test()
	