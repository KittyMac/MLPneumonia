from __future__ import division

import os

# Note: we're trying out PlaidML
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import csv
import pydicom
import numpy as np
import os
import sys
import random
import os.path
from model import IMG_SIZE
from PIL import Image,ImageDraw
import cv2
import glob

import multiprocessing
from multiprocessing import Pool
from contextlib import closing

kPatientID = 0
kBoundsX = 1
kBoundsY = 2
kBoundsWidth = 3
kBoundsHeight = 4
kTarget = 5
kCropX = 6
kCropY = 7
kCropWidth = 8
kCropHeight = 9

exportCount = 0

gridSize = 120

def adjustImageLevels(imageData):
	return np.clip(imageData * 2.0 - 0.5, 0.0, 1.0)

def dcmFilePathForPatient(patient):
	return "../data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def npyFilePathForPatient(patient, hasPneumonia):
	return "train/%d.%s.npy" % (hasPneumonia,patient[kPatientID])

def saveImageFromPatient(dcmFilePath,patient,hasPneumonia):
	# - load the DCM of the patient
	dcmData = pydicom.read_file(dcmFilePath)
	imageData = dcmData.pixel_array.astype('float32') / 255
	
	outputPath = npyFilePathForPatient(patient, hasPneumonia)
	Image.fromarray((imageData * 255).reshape(1024,1024)).convert("RGB").save("%s.png" % (outputPath))

def preprocessImage(imageFile):
	imageData = cv2.imread(imageFile, 0)
	imageData = imageData.astype('float32') / 255
	imageData = cv2.resize(imageData, (IMG_SIZE[0],IMG_SIZE[1])).reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2])
	outputPath = imageFile[:-4]
	print("caching image: %s" % outputPath)
	np.save(outputPath, imageData)

def deleteFile(path):
	os.remove(path)

if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] == "extract":
			mode = "extract"
		elif sys.argv[1] == "preprocess":
			mode = "preprocess"
	
	if mode == "unknown":
		print("valid modes: extract, preprocess")
		exit(0)
	
	if mode == "preprocess":
		# remove all existing npy cache files
		allCaches1 = glob.glob("train/*.npy")
		allCaches2 = glob.glob("not_train/*.npy")
		with closing(Pool(processes=multiprocessing.cpu_count()//2)) as pool:
			pool.map(deleteFile, allCaches1)
			pool.map(deleteFile, allCaches2)
			pool.terminate()
		# convert all png images to npy and save them
		allImages1 = glob.glob("train/*.png")
		allImages2 = glob.glob("not_train/*.png")
		with closing(Pool(processes=multiprocessing.cpu_count()//2)) as pool:
			pool.map(preprocessImage, allImages1)
			pool.map(preprocessImage, allImages2)
			pool.terminate()
	
	if mode == "extract":
		# Run through some number of samples from the training set, convert them to pngs and put them
		# in the train folder.  Then manually go through the images and crop out the lungs in preparation
		# for training the classifier

		patientInfo = []
	
		# 0. Load the patient information csv
		with open("../data/stage_1_train_images.csv") as csv_file:
			patientInfo = list(csv.reader(csv_file))
			patientInfo.pop(0)
	
	
		# DEBUG: for sanity purposes, let's test on a smaller subset first
		patientInfo = patientInfo[:50]
	
				
		# 1. Run through all patients without penumonia and make a list with their image paths
		imagesOfPneumoniaFreePatients = []
	
		for patient in patientInfo:
			if patient[kTarget] == "0":
				imagesOfPneumoniaFreePatients.append(dcmFilePathForPatient(patient))
	
		# 2. Run through all patients with penumonia
		for patient in patientInfo:
			if patient[kTarget] == "1":
				# Save the image of positive pneumonia
				saveImageFromPatient(dcmFilePathForPatient(patient), patient, 1)
				
	