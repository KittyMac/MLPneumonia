from __future__ import division

import os

# Note: we're trying out PlaidML
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# Note: we're using code in both
import sys
sys.path.insert(0, './pneumonia_classification')
sys.path.insert(0, './pneumonia_localization')

import csv
import random

from data import kPatientID
from data import kBoundsX
from data import kBoundsY
from data import kBoundsWidth
from data import kBoundsHeight
from data import kTarget
from data import adjustImageLevels

from GeneticLocalization import GeneticLocalization

import model
from model import IMG_SIZE

from PIL import Image,ImageDraw
import pydicom
import numpy as np
import cv2

interactiveMode = True
cnnModel = None

# This is the workhorse method of this file. We examine the patient's xray from the supplied path and
# return the bounding boxes of any cases of penumonia discovered.  We do this using a the GeneticLocalization
# class.
def Examine(dcmFilePath):
	
	global cnnModel
	if cnnModel is None:
		cnnModel = model.createModel(True)
	
	boxes = []
	
	dcmData = pydicom.read_file(dcmFilePath)
	dcmImage = dcmData.pixel_array.astype('float32') / 255
	dcmImage = adjustImageLevels(dcmImage)
	
	while len(boxes) < 40:
		
		gl = GeneticLocalization(dcmImage,cnnModel,boxes,(IMG_SIZE[0],IMG_SIZE[1]))
		box = gl.findBox()
		if box is None:
			break
		boxes.append(box)
		
	
	print(boxes)
	
	return boxes


def calculateIntersectionOverUnion(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def predictBox(cnnModel,npImage,box):
	cropped = npImage[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
	input = cv2.resize(cropped, (IMG_SIZE[0],IMG_SIZE[1]))
	
	Image.fromarray(input.reshape((IMG_SIZE[0],IMG_SIZE[1])) * 255).show()
	
	output = cnnModel.predict(input.reshape(1,IMG_SIZE[0],IMG_SIZE[1], 1))
	print("prediction", output, box)
	

def dcmFilePathForTrainingPatient(patient):
	return "pneumonia_classification/data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def dcmFilePathForTrestingPatient(patient):
	return "pneumonia_classification/data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def GetAllPatientInfo():
	patientInfo = []
	with open("pneumonia_classification/data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo
	
def TestPatient(patientId):
	# load the specified patient from the training data, examine their DCM, and compare the results to see how we did
	print("Test()", patientId)
	allPatients = GetAllPatientInfo()
	
	truthBoxes = []
	for patient in allPatients:
		if patient[kPatientID] == patientId:
			xmin = float(patient[kBoundsX])
			ymin = float(patient[kBoundsY])
			xmax = xmin + float(patient[kBoundsWidth])
			ymax = ymin + float(patient[kBoundsHeight])
			truthBoxes.append( (xmin, ymin, xmax, ymax) )
	
	dcmFilePath = dcmFilePathForTrainingPatient([patientId])
	predictionBoxes = Examine(dcmFilePath)
	
	print("truthBoxes",truthBoxes)
	print("predictionBoxes",predictionBoxes)
	
	# print prediction results for the truth boxes
	
	
	# show the image with all boxes
	dcmData = pydicom.read_file(dcmFilePath)
	imageData = dcmData.pixel_array
	imageDataNormalized = dcmData.pixel_array / 255
	imageDataNormalized = adjustImageLevels(imageDataNormalized)
	
	sourceImg = Image.fromarray(imageData).convert("RGB")	
	draw = ImageDraw.Draw(sourceImg)
	
	for box in truthBoxes:
		draw.rectangle(box, outline="green")
		predictBox(cnnModel,imageDataNormalized,box)
		
	for box in predictionBoxes:
		draw.rectangle(box, outline="yellow")
		predictBox(cnnModel,imageDataNormalized,box)
	
	if interactiveMode == True:
		# show the results
		sourceImg.show()
	
	
	
		''''
		# ask if correct
		if len(predictionBoxes) > 0:
			print("IoU",calculateIntersectionOverUnion(truthBoxes[0], predictionBoxes[0]))
			text = raw_input("Is this correct? (y/n): ")
			if text != "y":
				print("saving bad sample...")
			
				num = len(predictionBoxes) + len(predictionBoxes)
				input = np.zeros((num,IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]), dtype='float32')
				output = np.zeros((num,1), dtype='float32')
			
				idx = 0
				for box in predictionBoxes:
					outputPath = "pneumonia_classification/train/0.manual.%s.%d" % (patientId,random.randint(0,31543456543))
				
					image = Image.fromarray(imageData).convert("L")
					image = image.crop(box)
					image = image.resize((IMG_SIZE[0],IMG_SIZE[1]), Image.ANTIALIAS)
					outputData = np.array(image).astype('float32').reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]) / 255
					print("caching image: %s" % outputPath)
					np.save(outputPath, outputData)
				
					np.copyto(input[idx],outputData)
					output[idx][0] = 0
					idx += 1
				
				for box in truthBoxes:
					image = Image.fromarray(imageData).convert("L")
					image = image.crop(box)
					image = image.resize((IMG_SIZE[0],IMG_SIZE[1]), Image.ANTIALIAS)
					outputData = np.array(image).astype('float32').reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]) / 255
					np.copyto(input[idx],outputData)
					output[idx][0] = 1
					idx += 1
			
				cnnModel.fit(input,output,batch_size=2,shuffle=True,epochs=1,verbose=1)
			
				cnnModel.save(model.MODEL_H5_NAME)
				cnnModel.save("pneumonia_classification/%s" % (model.MODEL_H5_NAME))
			
			
				TestPatient(patientId)
		'''

def TestRandomPatients(num):
	# load random patients from the training data and test them
	print("TestRandomPatients()", num)

def GenerateSubmission():
	# load all patients from the submission testing data, examine them, and report the results in a CSV files suitable for submission to Kaggle
	print("GenerateSubmission()")


def ProcessPatient(patient):
	pass

if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] == "one":
			mode = "one"
		elif sys.argv[1] == "all":
			mode = "all"
	
	
	patientInfo = []

	# 0. Load the patient information csv
	with open("../data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	
	if mode == "one":
		pass
	
	if mode == "all":
		pass

	