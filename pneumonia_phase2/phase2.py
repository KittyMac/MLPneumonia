from __future__ import division

import os

# Note: we're using code from phase 1, specifically the model
import sys
sys.path.insert(0, '../pneumonia_phase1')

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


def AdjustPatientImage(patient):
	
	cnnModel = model.createModel(True)
	
	dcmFilePath = dcmFilePathForTrainingPatient(patient)
	dcmData = pydicom.read_file(dcmFilePath)
	dcmImage = dcmData.pixel_array.astype('float32') / 255
	
	print("adjusting image at:", dcmFilePath)	
	gl = GeneticLocalization(dcmImage,cnnModel,(IMG_SIZE[0],IMG_SIZE[1]))
	
	box = gl.findBox()
	print("box", box)
	
	sourceImg = Image.fromarray(dcmImage * 255).convert("RGB")	
	draw = ImageDraw.Draw(sourceImg)
	draw.rectangle(box, outline="yellow")
		
	return dcmImage,dcmImage[int(box[1]):int(box[3]),int(box[0]):int(box[2])],sourceImg

	
def phase1DataManualFixPath(patient):
	return "../pneumonia_phase1/manual_train/%s.png" % (patient[kPatientID])

def phase1DataTrainingPath(patient, isGood):
	if isGood:
		return "../pneumonia_phase1/train/%s.png" % (patient[kPatientID])
	return "../pneumonia_phase1/not_train/%s.png" % (patient[kPatientID])

def dcmFilePathForTrainingPatient(patient):
	return "../data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def dcmFilePathForTestingPatient(patient):
	return "../data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def GetAllPatientInfo():
	patientInfo = []
	with open("../data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo

if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["one", "all", "hot"]:
			mode = sys.argv[1]
		
	allPatients = GetAllPatientInfo()
	
	if mode == "one":
		AdjustPatientImage(random.choice(allPatients))
	
	if mode == "hot":
		while True:
			patient = random.choice(allPatients)
			fullImage,croppedImage,boxImage = AdjustPatientImage(patient)
			
			croppedImagePIL = Image.fromarray(croppedImage * 255).convert("RGB")
			fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
			#srcImage.show()
			
			boxImage.show()
			
			good = raw_input("Good? (Y)es/(N)o/(S)kip:")
			if good.lower() != "s":
				path = phase1DataTrainingPath(patient, good.lower() == "y")
				croppedImagePIL.save(path)
				print("saved to " + path)
				
				if good.lower() == "n":
					fullImagePIL.save(phase1DataManualFixPath(patient))
			
			cont = raw_input("Continue? Y/N:")
			if cont.lower() == "y":
				continue
			break
				
		
	
	if mode == "all":
		for patient in allPatients:
			AdjustPatientImage(patient)

	