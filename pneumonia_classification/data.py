from __future__ import division

import os

# Note: we're trying out PlaidML
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import csv
import pydicom
import numpy as np
import os
import random
import os.path
from model import IMG_SIZE
from PIL import Image,ImageDraw
import cv2

kPatientID = 0
kBoundsX = 1
kBoundsY = 2
kBoundsWidth = 3
kBoundsHeight = 4
kTarget = 5

exportCount = 0

def dcmFilePathForPatient(patient):
	return "data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def npyFilePathForPatient(patient, hasPneumonia):
	global exportCount
	exportCount += 1
	return "train/%d.%s.%d.npy" % (hasPneumonia,patient[kPatientID],exportCount)

def saveExtractedImageFromPatient(dcmFilePath,patient,hasPneumonia):
	# - load the DCM of the patient
	dcmData = pydicom.read_file(dcmFilePath)
	imageData = dcmData.pixel_array.astype('float32') / 255
				
	# - extract the bounded area
	xmin = float(patient[kBoundsX])
	ymin = float(patient[kBoundsY])
	xmax = xmin + float(patient[kBoundsWidth])
	ymax = ymin + float(patient[kBoundsHeight])
		
	croppedImage = imageData[int(ymin):int(ymax),int(xmin):int(xmax)]
	
	# - resize to the model training size
	outputImage = cv2.resize(croppedImage, (IMG_SIZE[0],IMG_SIZE[1])).reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2])
	
	# - save the training sample to disk, using filename to specify it has pneumonia
	outputPath = npyFilePathForPatient(patient, hasPneumonia)
	print("caching image: %s" % outputPath)
	np.save(outputPath, outputImage)
	
	'''
	if random.random() < 0.1:
		# CONFIRM THIS IS WORKING RIGHT!!!
		Image.fromarray((outputImage * 255).reshape(IMG_SIZE[0],IMG_SIZE[1])).show()
		sourceImg = Image.fromarray(dcmData.pixel_array).convert("RGB")
		draw = ImageDraw.Draw(sourceImg)
		draw.rectangle((xmin,ymin,xmax,ymax), outline="green")
		sourceImg.show()
		exit(1)
	'''

if __name__ == '__main__':
	
	
	# 0. Load the patient information csv
	# 1. Run through all patients without penumonia and make a list with their image paths
	# 2. Run through all patients with penumonia
	#    - load the DCM of the pneumonia patient
	#    - extract the bounded area
	#    - resize to the model training size
	#    - save the training sample to disk, using filename to specify it has pneumonia
	#    - load a random DCM of a pneumonia without patient
	#    - extract the bounded area
	#    - resize to the model training size
	#    - save the training sample to disk, using filename to specify it does not have pneumonia

	patientInfo = []
	
	# 0. Load the patient information csv
	with open("data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
				
	# 1. Run through all patients without penumonia and make a list with their image paths
	imagesOfPneumoniaFreePatients = []
	
	for patient in patientInfo:
		if patient[kTarget] == "0":
			imagesOfPneumoniaFreePatients.append(dcmFilePathForPatient(patient))
	
	# 2. Run through all patients with penumonia
	for patient in patientInfo:
		if patient[kTarget] == "1":
			# Save the image of positive pneumonia
			saveExtractedImageFromPatient(dcmFilePathForPatient(patient), patient, 1)
			# Save the image of negative pneumonia
			saveExtractedImageFromPatient(random.choice(imagesOfPneumoniaFreePatients), patient, 0)
	
	