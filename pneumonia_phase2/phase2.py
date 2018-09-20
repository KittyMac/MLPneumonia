from __future__ import division

import os

# Note: we're using code from phase 1, specifically the model
import sys
sys.path.insert(0, '../pneumonia_phase1')

import csv
import random
import glob

from data import kPatientID
from data import kBoundsX
from data import kBoundsY
from data import kBoundsWidth
from data import kBoundsHeight
from data import kTarget
from data import kCropX
from data import kCropY
from data import kCropWidth
from data import kCropHeight
from data import adjustImageLevels

from keras import backend as keras

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
	
	keras.clear_session()
	
	cnnModel = model.createModel(True)
	
	dcmFilePath = dcmFilePathForTrainingPatient(patient)
	dcmData = pydicom.read_file(dcmFilePath)
	dcmImage = dcmData.pixel_array.astype('float32') / 255
	
	print("adjusting image at:", dcmFilePath)	
	gl = GeneticLocalization(dcmImage,cnnModel,(IMG_SIZE[0],IMG_SIZE[1]))
	
	box = gl.findBox()
	print("box", box)
	
	sourceImg = None
	if box is not None:
		sourceImg = Image.fromarray(dcmImage * 255).convert("RGB")	
		draw = ImageDraw.Draw(sourceImg)
		draw.rectangle(box, outline="yellow")
		return dcmImage,dcmImage[int(box[1]):int(box[3]),int(box[0]):int(box[2])],sourceImg,box
	
	return dcmImage,dcmImage,sourceImg,box


def phase3PatientsPath():
	return "../pneumonia_phase3/stage_1_train_images.csv"
	
def phase3TestingPath(patient):
	return "../pneumonia_phase3/stage_1_test_images/%s.png" % (patient[kPatientID])

def phase3TrainingPath(patient):
	return "../pneumonia_phase3/stage_1_train_images/%s.png" % (patient[kPatientID])

def phase1DataManualFixPath(patient):
	return "../pneumonia_phase1/manual_train/%s.png" % (patient[kPatientID])

def phase1DataTrainingPath(patient, isGood):
	if isGood:
		return "../pneumonia_phase1/train/%s.png" % (patient[kPatientID])
	return "../pneumonia_phase1/not_train/%s.png" % (patient[kPatientID])

def dcmFilePathForTrainingPatient(patient):
	return "../data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def dcmFilePathForTestingPatient(patient):
	return "../data/stage_1_test_images/%s.dcm" % (patient[kPatientID])

def GetAllPatientInfo():
	patientInfo = []
	with open("../data/stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo

def GetExistingPhase3PatientInfo():
	patientInfo = []
	with open(phase3PatientsPath()) as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo

def GetPatientByID(patientID, allPatients):
	patientEntries = []
	for patient in allPatients:
		if patient[kPatientID] == patientID:
			patientEntries.append(patient)
	return patientEntries

if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["one", "all", "hot", "prepare3", "reset3"]:
			mode = sys.argv[1]
		
	allPatients = GetAllPatientInfo()
	
	if mode == "reset3":
		# clear the files in phase 3 dirs so we can extract from scratch
		allFiles = glob.glob("../pneumonia_phase3/stage_1_train_images/*.png")
		for file in allFiles:
			os.remove(file)
		allFiles = glob.glob("../pneumonia_phase3/stage_1_test_images/*.png")
		for file in allFiles:
			os.remove(file)
		with open(phase3PatientsPath(), mode='w') as patientFile:
			patientWriter = csv.writer(patientFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			patientWriter.writerow(['patientId', 'x', 'y', 'width', 'height', "Target", "cropX", "cropY", "cropWith", "cropHeight"])
		
		
	if mode == "prepare3":
		# prepare a new repopsitory of data for stage3 training.
		# 0. for all patients, read in the DCM, find the cropped lungs, export the cropped image as png
		# 1. add the crop box to the patient
		# 2. export a new stage_1_train_images.csv file which contains the crop box
		# Phase 3 will need all of this information in order to convert the pneumonia boxes to
		# the cropped image and back again.
		
		# you can specify the number of patients to try on the command line or
		# you can specify a specific patient by their patientid
		# you can specify nothing and all patients will be processed
		onlyThisPatientId = None
		
		num = len(allPatients)
		if len(sys.argv) >= 3:
			try:
				num = int(sys.argv[2])
			except:
				num = 1
				onlyThisPatientId = sys.argv[2]
		
		# load the CSV of patients we've already processed
		phase3Patients = GetExistingPhase3PatientInfo()
		
		# create a list of patientIds that we want to process.  Start by choosing
		# unique patientIds from allPatients
		patientsToProcess = {}
		if onlyThisPatientId is not None:
			# if we're only on the docket to do one patient, just do one patient...
			patientsToProcess[onlyThisPatientId] = 1
			
			# we also need to remove any old calculations for this patientID from phase3Patients
			for patient in phase3Patients[:]:
				if patient[kPatientID] == onlyThisPatientId:
					phase3Patients.remove(patient)
		else:
			# Otherwise grab all the unique patients in allPatients
			for patient in allPatients:
				patientsToProcess[patient[kPatientID]] = 1
			
			# and remove any of the patients we've already processed
			print("%d existing phase 3 patients" % (len(phase3Patients)))
			for patient in phase3Patients:
				try:
					patientsToProcess.pop(patient[kPatientID])
				except:
					pass
		
		print("%d patients available to process" % (len(patientsToProcess)))
		
		# get the patient ids to process as a list
		patientsToProcess = list(patientsToProcess.keys())

		# shuffle the patients around, then clip it to the number of patients
		# we are supposed to process
		random.shuffle(patientsToProcess)
		patientsToProcess = patientsToProcess[0:num]
		
		print("%d patients chosen to process for phase 3" % (len(patientsToProcess)))
		
		for patientId in patientsToProcess:
			
			patientEntries = GetPatientByID(patientId, allPatients)
			
			fullImage,croppedImage,boxImage,box = AdjustPatientImage(patientEntries[0])
			if box == None:
				print("ERROR: unable to detect lungs for patient. Saving image to phase 1 manual training...", patient[kPatientID])
				fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
				fullImagePIL.save(phase1DataManualFixPath(patient))
				continue
			
			croppedImagePIL = Image.fromarray(croppedImage * 255).convert("RGB")
			croppedImagePIL.save(phase3TrainingPath(patientEntries[0]))
			
			for patient in patientEntries:
				patient.append(box[0])
				patient.append(box[1])
				patient.append(box[2] - box[0])
				patient.append(box[3] - box[1])
				phase3Patients.append(patient)
		
		with open(phase3PatientsPath(), mode='w') as patientFile:
			patientWriter = csv.writer(patientFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			
			patientWriter.writerow(['patientId', 'x', 'y', 'width', 'height', "Target", "cropX", "cropY", "cropWith", "cropHeight"])
			for patient in phase3Patients:
				patientWriter.writerow(patient)
		
		
	
	if mode == "one" and len(sys.argv) >= 3:
		# getting stuck on local minimum: 59903052-140b-4065-aa4a-97599a3d0006
		patientID = sys.argv[2]
		for patient in allPatients:
			if patient[kPatientID] == patientID:
				fullImage,croppedImage,boxImage,box = AdjustPatientImage(patient)
				if box == None:
					print("ERROR: unable to detect lungs for patient. Saving image to phase 1 manual training...", patient[kPatientID])
					fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
					fullImagePIL.save(phase1DataManualFixPath(patient))
					continue
				
				croppedImagePIL = Image.fromarray(croppedImage * 255).convert("RGB")
				fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
				
				boxImage.show()
				
				good = raw_input("Good? (Y)es/(N)o/(S)kip:")
				if good.lower() != "s":
					path = phase1DataTrainingPath(patient, good.lower() == "y")
					croppedImagePIL.save(path)
					print("saved to " + path)
				
					if good.lower() == "n":
						fullImagePIL.save(phase1DataManualFixPath(patient))
				
				exit(0)
	
	if mode == "hot":
		while True:
			patient = random.choice(allPatients)
			fullImage,croppedImage,boxImage,box = AdjustPatientImage(patient)
			if box == None:
				print("ERROR: unable to detect lungs for patient. Saving image to phase 1 manual training...", patient[kPatientID])
				fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
				fullImagePIL.save(phase1DataManualFixPath(patient))
				continue
			
			croppedImagePIL = Image.fromarray(croppedImage * 255).convert("RGB")
			fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
			
			boxImage.show()
			
			good = raw_input("Good? (Y)es/(N)o/(S)kip:")
			if good.lower() != "s":
				path = phase1DataTrainingPath(patient, good.lower() == "y")
				croppedImagePIL.save(path)
				print("saved to " + path)
				
				if good.lower() == "n":
					fullImagePIL.save(phase1DataManualFixPath(patient))
			
			# remove it from the list so we don't see it again this run through
			allPatients.remove(patient)
			
			cont = raw_input("Continue? Y/N:")
			if cont.lower() == "y":
				continue
			break
				
		
	
	if mode == "all":
		for patient in allPatients:
			AdjustPatientImage(patient)

	