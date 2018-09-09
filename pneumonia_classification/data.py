from __future__ import division

# Note: we're trying out PlaidML
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import csv
import pydicom
import numpy as np
import os
import random
import os.path
from model import IMG_SIZE
from PIL import Image,ImageDraw

kPatientID = 0
kBoundsX = 1
kBoundsY = 2
kBoundsWidth = 3
kBoundsHeight = 4
kTarget = 5

def dcmFilePathForPatient(patient):
	return "data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def npyFilePathForPatient(patient, hasPneumonia):
	return "train/%d.%s.npy" % (hasPneumonia,patient[kPatientID])

def saveExtractedImageFromPatient(dcmFilePath,patient,hasPneumonia):
	# - load the DCM of the patient
	dcmData = pydicom.read_file(dcmFilePath)
	image = Image.fromarray(dcmData.pixel_array).convert("L")
				
	# - extract the bounded area
	left = float(patient[kBoundsX])
	top = float(patient[kBoundsY])
	right = left + float(patient[kBoundsWidth])
	bottom = top + float(patient[kBoundsHeight])
	
	image = image.crop((left,top,right,bottom))
	
	# - resize to the model training size
	image = image.resize((IMG_SIZE[0],IMG_SIZE[1]), Image.ANTIALIAS)
	imageData = np.array(image).astype('float32').reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]) / 255
	
	# - save the training sample to disk, using filename to specify it has pneumonia
	outputPath = npyFilePathForPatient(patient, hasPneumonia)
	print("caching image: %s" % outputPath)
	np.save(outputPath, imageData)

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
	
	