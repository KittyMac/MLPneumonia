from __future__ import division

from keras import backend as keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

import pydicom
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
import os

import signal
import time
import coremltools

from data import kPatientID
from data import dcmFilePathForTestingPatient
from data import dcmFilePathForTrainingPatient
from data import minMaxCropBoxForPatient

from model import IMG_SIZE

from PIL import Image,ImageDraw

ignoreSamples = [
	
	# I'm lazy and am holding off on these
	"a0930be8-f005-4141-a181-c1494cba358f",
	"850b9f56-6ae3-49bb-9cca-fb6cfe333b3f",
	"aebb193e-64ef-4368-b139-c0ee72709945",
	"494e613b-e81b-4ae0-856b-178c0310552b",
	"461aa799-2a1f-4710-a115-7410b088db06",
	"ebdce693-283b-462c-825e-a182c6063a85",
	"08ee8fda-45f1-4f46-a5ac-315a8d4dd196",
	"4e23d423-3ad6-4916-bfe7-67f60dda1e63",
	"8ae7e559-d1cf-4b54-a6bf-a653565540c3",
	"5eb5e5a8-076b-4bf8-ae80-26f6334e8790",
	"bddd048f-116f-467e-803f-fd234c53a61f",
	"ab6f5973-db3c-4c21-8981-189792ea16ee",
	
	
	# WTF bad images, ignore these
	"a4d719b4-04f5-4d58-86a2-b1ad3f1ba569", 
]

validationSamples = [
	
	# no pneumonia images
	"6ebb73c2-3b81-46cb-aa4a-8e52502dd0e1",
	"a7ab8ce9-d78b-469e-b596-57aca317e826",
	"56249696-a22f-441e-9e4e-8465d0ac7f70",
	"cb1bb35a-ddd4-4119-b866-6bc2642735e4",
	"5a09e3da-1f9d-4cda-84e9-d8bd5818daf9",
	"8daf7359-53ce-4607-9257-1d7e70ff1801",
	"4c05690d-628a-4ac5-8a41-6ee4e97a2663",
	"7b7ca7c2-9b4f-4370-9388-7ad0b2cdda63",
	
	# pneumonia images
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
	generator = data.DCMGenerator(False, validationSamples, ignoreSamples, True)
	
	# load the validation set
	vTrain, vLabel, vPatients = generator.validationImages()
	
	# number of images to generate
	n = 5000
	
	# Keep re-training on new sets, so we cycle in random non-pneumonia cases
	checkpoint = ModelCheckpoint(model.MODEL_H5_NAME, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	while True:
		Train(generator,vTrain,vLabel,_model,n,int(random.random()*5.0),checkpoint)
	
	_model.save(model.MODEL_H5_NAME)

def Train(generator,vTrain,vLabel,_model,n,epocs,checkpoint):
	train,label,patientIds = generator.generateImages(n, random.random())
	history = _model.fit(train,label,batch_size=32,shuffle=True,validation_data=(vTrain,vLabel),epochs=epocs,verbose=1,callbacks=[checkpoint])


def Test(patientID):
	_model = model.createModel(True)
	
	generator = data.DCMGenerator(False, None, ignoreSamples, False)
	
	if patientID is None:
		input,output,patientIds = generator.generateImages(64, 0.5)
	else:
		input,output = generator.generateImagesForPatient(patientID)
		patientIds = [patientID,patientID,patientID]
	
	results = _model.predict(input)
	
	
	for i in range(0,len(output)):
		
		if patientID is not None:
			
			# if we're only doing one patient, let's visually view the results to confirm they make sense
			patient = generator.patientForPatientID(patientID)
			dcmData = pydicom.read_file(dcmFilePathForTrainingPatient(patient))
			sourceImg = Image.fromarray(dcmData.pixel_array.reshape(1024,1024)).convert("RGB")
				
			draw = ImageDraw.Draw(sourceImg)
		
			cropBox = minMaxCropBoxForPatient(patient)
			draw.rectangle(cropBox, outline="yellow")
			
			boxes = generator.coordinatesFromOutput(output[i],IMG_SIZE)
			for box in boxes:
				adjustedBox = generator.convertBoxToSubmissionSize(patient, box)
				draw.rectangle(adjustedBox, outline="green")
			
			boxes = generator.coordinatesFromOutput(results[i],IMG_SIZE)
			for box in boxes:
				adjustedBox = generator.convertBoxToSubmissionSize(patient, box)
				draw.rectangle(adjustedBox, outline="red")
		
			sourceImg.show()
			sourceImg.save('/tmp/prediction_%s_t%s_p%s.png' % (patientIds[i], generator.convertOutputToString(output[i]), generator.convertOutputToString(results[i])))
			
		else:
			
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


def GenerateSubmission(patientID):
	_model = model.createModel(True)
	
	generator = data.DCMGenerator(True, None, ignoreSamples, False)
	
	patientsToProcess = []
	
	print("loading patients")
	if patientID is not None:
		for patient in generator.labelsInfo:
			if patient[kPatientID] == patientID:
				patientsToProcess.append(patient)
	else:
		for patient in generator.labelsInfo:
			patientsToProcess.append(patient)
	
	
	print("writing results file")
	outputFile = open("submission.csv", "w")
	outputFile.write("patientId,PredictionString\n")
	
	print( "predicting pnemonia on %d patients" % (len(patientsToProcess)) )
	for patient in patientsToProcess:
		print("  ... %s" % patient[kPatientID])
		
		input,output = generator.generateImagesForPatient(patient[kPatientID])
	
		results = _model.predict(input)
		
		boxes = generator.coordinatesFromOutput(results[0],IMG_SIZE)
		
		# Note: for the submission, all bounds must be reckoned in their non-cropped 1024x1024
		confidence = generator.convertOutputToString(results[0])
		if confidence >= 0.5:
			outputFile.write("%s," % (patient[kPatientID]))
			for box in boxes:
				adjustedBox = generator.convertBoxToSubmissionSize(patient, box)
				outputFile.write("%f %d %d %d %d " % (confidence,int(adjustedBox[0]),int(adjustedBox[1]),int(adjustedBox[2]-adjustedBox[0]),int(adjustedBox[3]-adjustedBox[1])))
					
			if patientID is not None:
				# if we're only doing one patient, let's visually view the results to confirm they make sense
				dcmData = pydicom.read_file(dcmFilePathForTestingPatient(patient))
				sourceImg = Image.fromarray(dcmData.pixel_array.reshape(1024,1024)).convert("RGB")
						
				draw = ImageDraw.Draw(sourceImg)
				
				cropBox = minMaxCropBoxForPatient(patient)
				draw.rectangle(cropBox, outline="yellow")
				
				for box in boxes:
					adjustedBox = generator.convertBoxToSubmissionSize(patient, box)
					draw.rectangle(adjustedBox, outline="red")
				sourceImg.show()
				
			
			outputFile.write("\n")
		else:
			outputFile.write("%s,\n" % patient[kPatientID])



if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["learn", "test", "submit"]:
			mode = sys.argv[1]
		
	if mode == "learn":
		Learn1()
	
	if mode == "submit":
		if len(sys.argv) >= 3:
			GenerateSubmission(sys.argv[2])
		else:
			GenerateSubmission(None)
	
	if mode == "test":
		if len(sys.argv) >= 3:
			Test(sys.argv[2])
		else:
			Test(None)
	
	
	