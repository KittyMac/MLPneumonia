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

# TODO: detect when an xray is inverted and fix
def FixInvertedImage(dcmImage, patient):
	if patient[kPatientID] == "037e120a-24fa-4f9c-8813-111136e3c288":
		print("Detected inverted xray, fixing...")
		dcmImage = np.ones(dcmImage.shape) - dcmImage
	return dcmImage

def AdjustPatientImage(dcmFilePath, patient, weightedRandoms):
	
	keras.clear_session()
	
	cnnModel = model.createModel(True)
	
	dcmData = pydicom.read_file(dcmFilePath)
	dcmImage = dcmData.pixel_array.astype('float32') / 255
	
	dcmImage = FixInvertedImage(dcmImage, patient)
	
	print("adjusting image at:", dcmFilePath)	
	gl = GeneticLocalization(dcmImage,cnnModel,(IMG_SIZE[0],IMG_SIZE[1]),weightedRandoms)
	
	box = gl.findBox()
	print("box", box)
	
	sourceImg = None
	if box is not None:
		sourceImg = Image.fromarray(dcmImage * 255).convert("RGB")	
		draw = ImageDraw.Draw(sourceImg)
		draw.rectangle(box, outline="yellow")
		return dcmImage,dcmImage[int(box[1]):int(box[3]),int(box[0]):int(box[2])],sourceImg,box
	
	return dcmImage,dcmImage,sourceImg,box


def phase3SubmissionPatientsPath():
	return "../pneumonia_phase3/stage_1_test_images.csv"

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
		return "../pneumonia_phase1/train/%d.%s.png" % (int(random.random() * 1000), patient[kPatientID])
	return "../pneumonia_phase1/not_train/%d.%s.png" % (int(random.random() * 1000), patient[kPatientID])

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

def GetExistingPhase3SubmissionPatientInfo():
	patientInfo = []
	try:
		with open(phase3SubmissionPatientsPath()) as csv_file:
			patientInfo = list(csv.reader(csv_file))
			patientInfo.pop(0)
	finally:
		return patientInfo

def GetPatientByID(patientID, allPatients):
	patientEntries = []
	for patient in allPatients:
		if patient[kPatientID] == patientID:
			patientEntries.append(patient)
	return patientEntries

def GetBoundingBoxWeightedArrays():
	# run through all phase3 patients
	# get their crop boxes in xmin,ymin,xmax,ymax
	# create two numpy arrays per value, to be used for weighted randoms
	allPatients = GetExistingPhase3PatientInfo()
	
	maxValue = 2048
	
	xminCount = np.zeros((maxValue,1))
	yminCount = np.zeros((maxValue,1))
	xmaxCount = np.zeros((maxValue,1))
	ymaxCount = np.zeros((maxValue,1))
	
	xminList = []
	xminWeights = []
	xmaxList = []
	xmaxWeights = []
	yminList = []
	yminWeights = []
	ymaxList = []
	ymaxWeights = []
	
	for patient in allPatients:
		xmin = int(patient[kCropX])
		ymin = int(patient[kCropY])
		xmax = xmin + int(patient[kCropWidth])
		ymax = ymin + int(patient[kCropHeight])
		
		xminCount[xmin] += 1
		yminCount[ymin] += 1
		xmaxCount[xmax] += 1
		ymaxCount[ymax] += 1		
		
	xminSum = np.sum(xminCount)
	xmaxSum = np.sum(xmaxCount)
	yminSum = np.sum(yminCount)
	ymaxSum = np.sum(ymaxCount)
	
	for i in range(0,maxValue):
		if xminCount[i] > 0:
			xminList.append(i)
			xminWeights.append(xminCount[i][0] / xminSum)
		if xmaxCount[i] > 0:
			xmaxList.append(i)
			xmaxWeights.append(xmaxCount[i][0] / xmaxSum)
		if yminCount[i] > 0:
			yminList.append(i)
			yminWeights.append(yminCount[i][0] / yminSum)
		if ymaxCount[i] > 0:
			ymaxList.append(i)
			ymaxWeights.append(ymaxCount[i][0] / ymaxSum)
	
	weightedRandoms = {}
	
	xminListNP = np.empty(len(xminList), dtype=object)
	xminListNP[:] = xminList
	yminListNP = np.empty(len(xminList), dtype=object)
	yminListNP[:] = xminList
	xmaxListNP = np.empty(len(ymaxList), dtype=object)
	xmaxListNP[:] = ymaxList
	ymaxListNP = np.empty(len(ymaxList), dtype=object)
	ymaxListNP[:] = ymaxList
	
	xminWeightsNP = np.empty(len(xminWeights), dtype=float)
	xminWeightsNP[:] = xminWeights
	yminWeightsNP = np.empty(len(xminWeights), dtype=float)
	yminWeightsNP[:] = xminWeights
	xmaxWeightsNP = np.empty(len(ymaxWeights), dtype=float)
	xmaxWeightsNP[:] = ymaxWeights
	ymaxWeightsNP = np.empty(len(ymaxWeights), dtype=float)
	ymaxWeightsNP[:] = ymaxWeights
	
	weightedRandoms["xmin"] = (xminListNP, xminWeightsNP)
	weightedRandoms["ymin"] = (yminListNP, yminWeightsNP)
	weightedRandoms["xmax"] = (xmaxListNP, xmaxWeightsNP)
	weightedRandoms["ymax"] = (ymaxListNP, ymaxWeightsNP)
	
	return weightedRandoms


if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["one", "all", "hot", "prepare3", "submission3", "submissionOne", "reset3"]:
			mode = sys.argv[1]
	
	
	weightedRandoms = GetBoundingBoxWeightedArrays()
	
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
	
	
	if mode == "submissionOne" and len(sys.argv) >= 3:
		patientID = sys.argv[2]
		patient = [patientID, 0, 0, 0, 0, 0]
		
		fullImage,croppedImage,boxImage,box = AdjustPatientImage(dcmFilePathForTestingPatient(patient), patient, weightedRandoms)
		if box == None:
			print("ERROR: unable to detect lungs for patient. Saving image to phase 1 manual training...", patient[kPatientID])
			fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
			fullImagePIL.save(phase1DataManualFixPath(patient))
			exit(0)
		
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
	
	if mode == "submission3":
		# this is the same as prepare3, but processes the competition test set
		phase3Patients = GetExistingPhase3SubmissionPatientInfo()
		allSubmissionPatients = []
		allFiles = glob.glob("../data/stage_1_test_images/*.dcm")
		
		onlyThisPatientId = None
		if len(sys.argv) >= 3:
			onlyThisPatientId = sys.argv[2]
		
		# 1. create fake patient information
		for file in allFiles:
			patient = [os.path.splitext(os.path.basename(file))[0], 0, 0, 0, 0, 0]
			
			if onlyThisPatientId == None:
				# normal operation, check if patient has been processed before. if so, don't process
				patientAlreadyExists = False
				for otherPatient in phase3Patients:
					if otherPatient[kPatientID] == patient[kPatientID]:
						patientAlreadyExists = True
						break
				
				# otherwise, we haven't done this patient before so we need to process him
				if patientAlreadyExists == False:
					allSubmissionPatients.append(patient)
			else:
				# special operation, we want to do onlyThisPatientId regardless if we've done him before
				if onlyThisPatientId == patient[kPatientID]:
					# remove all previous mentions of this patient, then add them
					for otherPatient in phase3Patients[:]:
						if onlyThisPatientId == otherPatient[kPatientID]:
							phase3Patients.remove(otherPatient)
					allSubmissionPatients.append(patient)
		
		# 2. process the patients
		print("%d submission patients to process for phase 3" % (len(allSubmissionPatients)))
		
		for patient in allSubmissionPatients:			
			fullImage,croppedImage,boxImage,box = AdjustPatientImage(dcmFilePathForTestingPatient(patient), patient, weightedRandoms)
			if box == None:
				print("ERROR: unable to detect lungs for patient. Saving image to phase 1 manual training...", patient[kPatientID])
				fullImagePIL = Image.fromarray(fullImage * 255).convert("RGB")
				fullImagePIL.save(phase1DataManualFixPath(patient))
				continue
			
			croppedImagePIL = Image.fromarray(croppedImage * 255).convert("RGB")
			croppedImagePIL.save(phase3TestingPath(patient))
			
			patient.append(box[0])
			patient.append(box[1])
			patient.append(box[2] - box[0])
			patient.append(box[3] - box[1])
			
			phase3Patients.append(patient)
			
			# save the .csv file after every patient just in case we don't finish processing
			with open(phase3SubmissionPatientsPath(), mode='w') as patientFile:
				patientWriter = csv.writer(patientFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			
				patientWriter.writerow(['patientId', 'x', 'y', 'width', 'height', "Target", "cropX", "cropY", "cropWith", "cropHeight"])
				for patient in phase3Patients:
					patientWriter.writerow(patient)
		
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
			
			fullImage,croppedImage,boxImage,box = AdjustPatientImage(dcmFilePathForTrainingPatient(patientEntries[0]), patientEntries[0], weightedRandoms)
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
				fullImage,croppedImage,boxImage,box = AdjustPatientImage(dcmFilePathForTrainingPatient(patient), patient, weightedRandoms)
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
			fullImage,croppedImage,boxImage,box = AdjustPatientImage(dcmFilePathForTrainingPatient(patient), patient, weightedRandoms)
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
			AdjustPatientImage(dcmFilePathForTrainingPatient(patient), patient, weightedRandoms)

	