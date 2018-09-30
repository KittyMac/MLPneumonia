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
	
	"b957974c-211b-4f52-86d3-25d18825dfa3",
	"49c04987-96af-4edb-b560-53c56a357cac",
	"0fe227eb-592b-4ae5-b050-e7573d423953",
	"01fa0f5e-00c3-41cb-b5c7-10600c8633ae",
	"02e93c9a-c063-4bf2-9fa1-cb26692d58f8",
	"3abb7176-035d-46cc-844e-820870e8154b",
	"3c239733-d999-4dad-ab3b-3b5bf7b893b7",
	"3e2db6da-3b64-4388-8098-1c4c037ec03a",
	"3fad6559-5289-4da1-a682-df0c97a08e38",
	"4e23d423-3ad6-4916-bfe7-67f60dda1e63",
	"5a7cc989-874a-45e2-bdb3-9967f331babd",
	"5cb11c59-6ba7-4f94-b9f8-12927ffb8981",
	"5eb5e5a8-076b-4bf8-ae80-26f6334e8790",
	"5f8240fa-94df-47ed-8f8c-3eca1f92bc28",
	"06d5a58d-baf1-4937-bfc3-00db1fb2b1be",
	"7b43d32e-133e-46f6-8698-6767d0b48d5e",
	"7f29e031-874c-4ea2-9f7f-edc056cb5e83",
	"8a8be155-755a-48f9-a4aa-f1fd0e1e263b",
	"8ae7e559-d1cf-4b54-a6bf-a653565540c3",
	"8d469ff7-e46f-4102-9e95-fdd63753b34c",
	"08ee8fda-45f1-4f46-a5ac-315a8d4dd196",
	"08f17660-dc61-4a5d-b6fb-10dacc1e20a2",
	"9b788ef2-8721-4f1f-9d4d-8a2e0d65cba4",
	"09bfe52b-201a-435a-b04f-ada40a64b5b5",
	"9cbf0728-d245-4782-8aca-c25c5e8e158f",
	"12e6144b-edc2-4d4e-9ae8-ca95d9aad065",
	"15b66169-1cbc-40a5-bbd6-3acbad9f6445",
	"31ad19e6-fe6d-40da-b2cc-a0dc5554d79e",
	"48ee747e-6d3b-4fbc-a9d7-f438600ba6de",
	"66ab31a6-be10-4f4e-ad4d-b4aeb9371271",
	"70d7f4df-5497-4205-a36b-e47b58145fa2",
	"72a6910e-c0f2-422e-988b-dcbf1450f0e6",
	"80b493e1-a3a6-47a2-ae68-6a502c081507",
	"80e7b516-3f0d-4f19-b7e4-7e2849ec0e13",
	"94fd8652-cb5a-40c6-9c5e-e429de9e8715",
	"308eab64-517b-46f1-8865-d9566b087bff",
	"314f6860-4382-4ccd-a787-34b388e673f1",
	"335d17b1-bdfb-4939-9dc9-fa93f09648eb",
	"371e6626-7884-4232-9629-0220fe6e7172",
	"461aa799-2a1f-4710-a115-7410b088db06",
	"494e613b-e81b-4ae0-856b-178c0310552b",
	"624a95b9-354b-415e-8bcc-382c6510469d",
	"819d6b1a-4e49-4fd6-8ac1-f7a9f355617e",
	"826a4142-b079-4825-85f7-5fd69292e4cf",
	"850a00db-8884-45d3-b3cf-1ba8635b9294",
	"850b9f56-6ae3-49bb-9cca-fb6cfe333b3f",
	"3874e3f6-0c79-4fc1-a874-1d761e475e53",
	"3993a64f-ab45-4594-abe9-2807e19a1d71",
	"4005e782-c12e-4657-9b29-1707ee6eb08a",
	"8804e7cf-a2d8-496b-92c6-140438823b07",
	"36946b92-64ac-4bfd-9984-e3f9b8b95bc5",
	"098435af-3834-466a-b17a-038ac4f0e464",
	"98833b28-3925-4098-bf44-c81a9e687838",
	"338907f5-8c8c-47e7-a57d-6c9eb5b4529b",
	"395738ff-53d2-42e6-aa6d-a01e4a07f01f",
	"568694ec-5807-407a-8c10-40d0831e367d",
	"573957bf-09b6-40d0-aca6-389696465e7b",
	"741500a4-d128-4b85-852b-428779f5c5ec",
	"809830cd-c757-4e81-abec-0d6d935ae0ce",
	"889672e3-66b7-4879-9b02-0bc41d3f66d2",
	"961041ed-849e-4253-8d6a-97a8aa29b2d4",
	"1703014f-cb4e-43c2-893d-288d087aafbd",
	"4068843a-d393-45cb-9d08-567238cdc6d2",
	"06089341-a69b-4777-817b-041be894b436",
	"41560731-6e3f-44b9-b951-e789bb8e0cc8",
	"78053750-8aa7-4755-a364-037d71a65426",
	"a0c64b6b-5c91-422e-96a6-5389d9c23cec",
	"a3dda87f-1087-42ee-b602-453eaac55e5f",
	"a3f5a3d9-78a1-4689-8b1e-15f146e14b4a",
	"a4ac5fc1-dbc7-4380-8265-04e05f959c0e",
	"a0930be8-f005-4141-a181-c1494cba358f",
	"a9006409-11f0-4d8d-acff-4f5e973547b1",
	"aa1970b8-5039-4ba9-a933-4bd96ad81a91",
	"ab6f5973-db3c-4c21-8981-189792ea16ee",
	"abff4044-3c0a-458f-b0c7-04254be7fc0c",
	"ac6b782f-206e-4cd2-8002-427c2843f90b",
	"ac70e5ac-f93d-4fbf-a8d8-6db501f56cc2",
	"ace3688f-2479-4468-9759-1e9a1fe02d6f",
	"aeb2a32c-595a-44c2-b827-c6ce0f6ff4f3",
	"aebb193e-64ef-4368-b139-c0ee72709945",
	"b0b3639e-7625-4bd2-b9a6-0bb87918987a",
	"b3bd974d-ec10-45e4-a9b0-0c72cb969f7e",
	"b6b3152f-ef3f-41ee-b015-118500c0f08a",
	"b020af54-1c80-4f36-829e-a5c9515802fd",
	"b35d57ee-f22b-4c8c-b959-3ee8eecef555",
	"bb7dca62-6180-4e52-81b6-8a9e09a4bfae",
	"bbe15fbc-e79c-4206-a901-5b23888f8a8b",
	"bc9683bf-595d-497e-8c99-ac9baed1a2eb",
	"bd7b6626-82a0-4410-b96b-3109223128e1",
	"bddd048f-116f-467e-803f-fd234c53a61f",
	"bfd795ae-7656-40fc-b44e-566515e52ed2",
	"c10f3035-c0f4-4436-a6f9-bbeb87777231",
	"c98faad9-6dd2-45fe-87af-63bbb94eeb77",
	"c2016b1b-e24d-4eaf-af0f-14cf6f0bd9ad",
	"c0018701-b726-4c75-b5fb-686be2d8b00d",
	"cb3816db-93c9-4d0a-a163-eb40363c45fe",
	"cd4f439d-61be-4ecb-bc27-383f9cd29569",
	"cd064f36-a22d-4614-a4c5-96c2d36bbe7c",
	"ce97b66d-f5d4-490e-8a6f-2b926011eae7",
	"d4b3478b-d430-4c89-95b5-1c1c27c8eb71",
	"d5e1c771-f2f2-4a2a-afbd-4b14b3198500",
	"d50e7c06-768f-4943-8e00-392d664fe580",
	"d979af2c-4ed0-42da-9b2d-5cd3733b8d13",
	"da69ffb8-40fa-49ea-b11e-9ddff40845e8",
	"db4c7637-7784-4031-b983-4692715525c5",
	"dd830a3b-db9e-42bd-8d3a-8d52f40ddbde",
	"e5bff349-8b02-4c62-8c64-b32344fa690c",
	"eb10b7c7-fc94-488a-a5a9-704876d78ebb",
	"ebd9a666-b3ff-407c-9d50-9af3a285304f",
	"ebdce693-283b-462c-825e-a182c6063a85",
	"f2e71eff-2656-4b52-a014-eea3d0d00efc",
	"f218fda3-c484-4836-867d-b7ad037bfe4e",
	"f957afbf-fa78-47a0-942c-17b2f5262fc9",
	
	
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
	checkpoint = ModelCheckpoint(model.MODEL_H5_NAME, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	while True:
		Train(generator,vTrain,vLabel,_model,n,7,checkpoint)
	
	_model.save(model.MODEL_H5_NAME)

def Train(generator,vTrain,vLabel,_model,n,epocs,checkpoint):
	train,label,patientIds = generator.generateImages(n, 0.3 + (random.random() * 0.6))
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
	
	
	