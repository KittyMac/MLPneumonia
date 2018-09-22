from __future__ import division

from keras import backend as keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

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

from model import IMG_SIZE

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


validationSamples = [
	"a6236ddd-6367-4569-b5f1-07d2df9390ad",
	"3cbd12c8-6302-43b1-a28c-fcc892540c2e",
	"359355a8-5981-406a-a8c3-3dcfdde9169e",
	"f02fe6d5-c4ac-4c83-9d9d-5274eb3488d5",
	"34d340bd-2928-41cb-8d2a-98ba84999a01",
	"7c5188fb-01b8-4923-9715-83f13a5e861b",
	"de304f9b-f407-43d5-89b9-bdb07f8308b1",
	"3b66da46-f0a2-4e41-bd6d-68a8ba799708",
]


def Learn1():
	
	# New tests:
	# 0. Baseline (no augmentation, 0.5 split, old model, no duplicates)
	# 1. Only train on postitive samples
	# 2. Auto-levels images in preprocess stage
	# 3. Less conv2d more dense layers
	
	
	print("Learning Phase 1")
	
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)
	
	# 2. train the model
	print("initializing the generator")
	generator = data.DCMGenerator(False, validationSamples, False)
	
	# number of images to generate
	n = 5000
	
	# train hard for a long time on this sample set
	Train(generator,_model,n,100)
	
	_model.save(model.MODEL_H5_NAME)

def Train(generator,_model,n,epocs):
	checkpoint = ModelCheckpoint(model.MODEL_H5_NAME, monitor='loss', verbose=0, save_best_only=True, mode='min')
	train,label,patientIds = generator.generateImages(n, 0.5)
	_model.fit(train,label,batch_size=128,shuffle=True,epochs=epocs,verbose=1,callbacks=[checkpoint])

# Load smaller chunks of samples to maximize data augmentation. Use an SGD with a low
# weight so we can "fine tune" into the data augmentation
def Learn2():
		
	# 1. create the model
	print("creating the model")
	_model = model.createModel(True)

	# 2. train the model
	print("initializing the generator")
	generator = data.DCMGenerator(False, validationSamples, False)
	
	iterations = 1000000
		
	print("beginning training")	
	handler = SignalHandler()
	i = 0
	while True:
		
		if handler.stop_processing:
			break
		
		n = 5000
		print(i)
		Train(generator,_model,n,1)
		i += n
		
		if i >= iterations:
			break
	
	_model.save(model.MODEL_H5_NAME)






def Test(patientID):
	_model = model.createModel(True)
	
	generator = data.DCMGenerator(False, None, False)
	
	if patientID is None:
		input,output,patientIds = generator.generateImages(64, 0.5)
	else:
		input,output = generator.generateImagesForPatient(patientID)
		patientIds = [patientID,patientID,patientID]
	
	results = _model.predict(input)
	
	
	for i in range(0,len(output)):
		sourceImg = Image.fromarray(input[i].reshape(IMG_SIZE[0],IMG_SIZE[1]) * 255.0).convert("RGB")
		
		draw = ImageDraw.Draw(sourceImg)
		
		boxes = generator.coordinatesFromOutput(output[i],IMG_SIZE)
		for box in boxes:
			draw.rectangle(box, outline="green")
			
		boxes = generator.coordinatesFromOutput(results[i],IMG_SIZE)
		for box in boxes:
			draw.rectangle(box, outline="red")
		
		sourceImg.save('/tmp/prediction_%s_t%s_p%s.png' % (patientIds[i], generator.convertOutputToString(output[i]), generator.convertOutputToString(results[i])))
		
		if len(output) == 1:
			sourceImg.show()

'''
def GenerateSubmission():
	_model = model.createModel(True)
	
	print("loading submission images")
	generator = data.DCMGenerator("data/stage_1_test_images", None)
	patients,input = generator.generatePredictionImages()
	
	print("predicting pnemonia")
	results = _model.predict(input)
	
	print("writing results file")
	outputFile = open("submission.csv", "w")
	outputFile.write("patientId,PredictionString\n")
	for i in range(0,len(patients)):
		print("  ... %s" % patients[i][0])
		
		# Note: for the submission, all bounds must be reckoned in 1024x1024, the size of the samples provided
		boxes = generator.coordinatesFromOutput(results[i],(1024,1024))
		confidence = generator.convertOutputToString(results[i])
		if confidence >= 0.5:
			outputFile.write("%s," % (patients[i][0]))
			for box in boxes:
				outputFile.write("%f %d %d %d %d " % (confidence,box[0],box[1],box[2]-box[0],box[3]-box[1]))
			outputFile.write("\n")
		else:
			outputFile.write("%s,\n" % patients[i][0])
'''


if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["learn", "test"]:
			mode = sys.argv[1]
		
	if mode == "learn":
		Learn1()
	
	if mode == "test":
		if len(sys.argv) >= 3:
			Test(sys.argv[2])
		else:
			Test(None)
	
	
	