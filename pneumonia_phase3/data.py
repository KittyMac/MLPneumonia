# script responsible for generating audio sample of specific sounds waves.  Supports
# exporting those samples to WAV files for confirmation.

from __future__ import division

import csv
import pydicom
import numpy as np
import os
import random
import os.path
import sys
import glob
import cv2
from model import IMG_SIZE
from model import IMG_SUBDIVIDE
from PIL import Image,ImageDraw

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

kMaxImageOffset = 4

kThreshold = 0.5

class DCMGenerator():
	
	def __init__(self,finalSubmission,validationSamples,shouldAugment):
		self.shouldAugment = shouldAugment
		if finalSubmission == True:
			self.directory = "stage_1_test_images/"
			self.labelsInfo = []
			allFiles = glob.glob("stage_1_train_images/*.npy")
			for file in allFiles:
				patient = [os.path.splitext(file)[0], 0, 0, 0, 0, 0, 0, 0, 0, 0]
				self.labelsInfo.append(patient)
		else:
			self.directory = "stage_1_train_images/"
			self.labelsInfo = GetPhase3PatientInfo()
			if validationSamples is not None:
				for patient in self.labelsInfo[:]:
					if patient[kPatientID] in validationSamples:
						print("removing patient for validation: %s" % (patient[kPatientID]))
						self.labelsInfo.remove(patient)
	
	def numberOfSamples(self):
		return len(self.labelsInfo)
	
	def simpleImageAugment(self,imageData,xOff,yOff):
		# very simple slide x/y image augmentation
		imageData = np.roll(imageData, int(xOff), axis=1)
		imageData = np.roll(imageData, int(yOff), axis=0)
		return imageData
		
	
	def loadImageForPatientId(self,patient):
		imageData = None
		
		patientId = patient[kPatientID]
				
		imageFilePath = self.directory + "/" + patientId + ".npy"
		try:
			return np.load(imageFilePath)
		except:
			print("preprocessed file does not exist: ", imageFilePath)
		
		exit(1)
		
	
	def generateImages(self,num,positiveSplit):
		
		# 1. we do not want duplicates
		# 2. we want to combine patients into a single image
		
		localLabelsInfo = self.labelsInfo[:]
						
		random.shuffle(localLabelsInfo)
		
		patientIds = []
		
		numPositive = 0
		numNegative = 0
		
		processedPatientList = []
		
		for i in range(0,num):
			
			if len(localLabelsInfo) == 0:
				break
			
			# random selection with attempt to balance the patient population
			attempts = 10000
			while attempts > 0:
				patient = random.choice(localLabelsInfo)
				if numPositive <= (numNegative+numPositive)*positiveSplit and patient[kTarget] == "1":
					break
				if numPositive > (numNegative+numPositive)*positiveSplit and patient[kTarget] == "0":
					break
				attempts -= 1
			
			if attempts <= 0:
				# we could not find any of the right kind of patients left to balance population, so end early
				break
			
			if patient[kTarget] == "1":
				numPositive += 1
			else:
				numNegative += 1
			
			# remove all references of this patient from localLabelsInfo so we do not duplicate
			for duplicatePatient in self.labelsInfo:
				if patient[kPatientID] == duplicatePatient[kPatientID]:
					localLabelsInfo.remove(duplicatePatient)
			
			patientIds.append(patient[kPatientID])
			
			input2,output2 = self.generateImagesForPatient(patient[kPatientID])
			
			processedPatientList.append( (patient, input2[0], output2[0]) )
		
		num = len(processedPatientList)
		
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((num,IMG_SUBDIVIDE+IMG_SUBDIVIDE), dtype='float32')
		
		for i in range(0,len(processedPatientList)):
			processedPatient = processedPatientList[i]
			np.copyto(input[i], processedPatient[1])
			np.copyto(output[i], processedPatient[2])
		
		print(numPositive, numNegative)
							
		return input,output,patientIds
	
	def generateImagesForPatient(self,patientID):
		
		localPatientInfo = []
		for i in range(0,len(self.labelsInfo)):
			patient = self.labelsInfo[i]
			if patient[kPatientID] == patientID:
				localPatientInfo.append(patient)

		
		input = np.zeros((1,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((1,IMG_SUBDIVIDE+IMG_SUBDIVIDE), dtype='float32')
		
		localPatient = localPatientInfo[0]
		
		xOffForImage = int(random.random() * (kMaxImageOffset*2) - kMaxImageOffset)
		yOffForImage = int(random.random() * (kMaxImageOffset*2) - kMaxImageOffset)
		
		if self.shouldAugment == False:
			xOffForImage = 0
			yOffForImage = 0

		imageData = self.loadImageForPatientId(localPatient)
		imageData = self.simpleImageAugment(imageData,xOffForImage,yOffForImage)
		np.copyto(input[0], imageData)
	
		if localPatient[kTarget] == "1":
			# note: we may have multiple data lines per patient, so we want to
			# combine their outputs such that there is only one combined training sample
			for patient in localPatientInfo:
				imgWidth = float(patient[kCropWidth])
				imgHeight = float(patient[kCropHeight])
				
				xOffForBounds = xOffForImage * (imgWidth / IMG_SIZE[0])
				yOffForBounds = yOffForImage * (imgHeight / IMG_SIZE[1])

				# this is the box at cropped image size & location
				box = minMaxCroppedBoxForPatient(patient,xOffForBounds,yOffForBounds)
		
				# Note: the "full size" of the image here is the size of the cropped image
				xdelta = (imgWidth / IMG_SUBDIVIDE)
				ydelta = (imgHeight / IMG_SUBDIVIDE)
				for x in range(0, IMG_SUBDIVIDE):
					for y in range(0, IMG_SUBDIVIDE):
						xValue = x * xdelta
						yValue = y * ydelta
						if xValue+xdelta >= box[0] and xValue <= box[2]:
							output[0][x] = 1
						if yValue+ydelta >= box[1] and yValue <= box[3]:
							output[0][IMG_SUBDIVIDE+y] = 1
				
		return input,output
	
	def generatePredictionImages(self):
		
		num = len(self.labelsInfo)
				
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
				
		for i in range(0,num):
			patient = self.labelsInfo[i]
			imageData = self.loadImageForPatientId(patient)
			np.copyto(input[i], imageData)
										
		return self.labelsInfo,input
	
	def convertOutputToString(self,output):
		for x in range(0,IMG_SUBDIVIDE):
			for y in range(0,IMG_SUBDIVIDE):
				if output[x] >= kThreshold and output[IMG_SUBDIVIDE+y] >= kThreshold:
					return 1
		return 0
	
	
	def identifyPeaksForAxis(self,output):
		IMG_SUBDIVIDE = int(len(output))
		xdelta = 1.0 / IMG_SUBDIVIDE
		ydelta = 1.0 / IMG_SUBDIVIDE
		
		peak_indentity = np.zeros((IMG_SUBDIVIDE), dtype='float32')
		minPeakSize = 3
		
		peakIdx = 0
		x = 0
		while x < IMG_SUBDIVIDE:
			xValue = (x*xdelta)
			
			# step forward until we find the first peak
			if output[x] >= kThreshold:
				# ensure this peak is large enough (ignore little peaks)
				isPeak = False
				for x2 in range(x,IMG_SUBDIVIDE):
					if output[x2] < kThreshold:
						if x2 - x > minPeakSize:
							isPeak = True
				
				# peak identified, fill it out
				if isPeak:
					peakIdx += 1
					for x2 in range(x,IMG_SUBDIVIDE):
						if output[x2] >= kThreshold:
							peak_indentity[x2] = peakIdx
						else:
							break
					x = x2
			x += 1
		return peak_indentity,peakIdx
	
	def coordinatesFromOutput(self,output,size):
		# 1. run through X and Y outputs and identify peaks (0 for not in peak, increasing numeral of different peak)
		# 2. run through X and Y output values, and identify new bounds based on peak index
		IMG_SUBDIVIDE = int(len(output)/2)
		xdelta = 1.0 / IMG_SUBDIVIDE
		ydelta = 1.0 / IMG_SUBDIVIDE
		
		x_output = output[0:IMG_SUBDIVIDE]
		y_output = output[IMG_SUBDIVIDE:]
		
		x_peaks,num_x_peaks = self.identifyPeaksForAxis(x_output)
		y_peaks,num_y_peaks = self.identifyPeaksForAxis(y_output)
		
		'''
		print("x axis")
		print(x_output)
		print(x_peaks)
		print("-----------------------")
		print("y axis")
		print(y_output)
		print(y_peaks)
		print("-----------------------")
		'''
		
		boxes = []
		
		for peakIdx in range(1,num_x_peaks+1):
		
			xmin = 1.0
			xmax = 0.0
			ymin = 1.0
			ymax = 0.0
	
			for x in range(0,IMG_SUBDIVIDE):
				for y in range(0,IMG_SUBDIVIDE):
					xValue = (x*xdelta)
					yValue = (y*ydelta)
					
					if x_peaks[x] == peakIdx:
						if output[x] >= kThreshold and output[IMG_SUBDIVIDE+y] >= kThreshold:
							if xValue < xmin:
								xmin = xValue
							if xValue > xmax:
								xmax = xValue
							if yValue < ymin:
								ymin = yValue
							if yValue > ymax:
								ymax = yValue
			
			# only boxes with decent width or height are counted
			box = (int(xmin*size[1]),int(ymin*size[0]),int(xmax*size[1]),int(ymax*size[0]))
			if box[2] - box[0] > 10 and box[3] - box[1] > 10:
				boxes.append(box)
				
		return boxes
	
def GetPhase3PatientInfo():
	patientInfo = []
	with open("stage_1_train_images.csv") as csv_file:
		patientInfo = list(csv.reader(csv_file))
		patientInfo.pop(0)
	return patientInfo

def dcmFilePathForTrainingPatient(patient):
	return "../data/stage_1_train_images/%s.dcm" % (patient[kPatientID])

def dcmFilePathForTestingPatient(patient):
	return "../data/stage_1_test_images/%s.dcm" % (patient[kPatientID])

def minMaxBoxForPatient(patient):
	x = float(patient[kBoundsX])
	y = float(patient[kBoundsY])
	w = float(patient[kBoundsWidth])
	h = float(patient[kBoundsHeight])
	return (x,y,x+w,y+h)
	
def minMaxCroppedBoxForPatient(patient,xOffForBounds,yOffForBounds):
	cx = float(patient[kCropX])
	cy = float(patient[kCropY])
	x = float(patient[kBoundsX])
	y = float(patient[kBoundsY])
	w = float(patient[kBoundsWidth])
	h = float(patient[kBoundsHeight])
	return ( (x-cx)+xOffForBounds, (y-cy)+yOffForBounds, ((x-cx)+xOffForBounds)+w, ((y-cy)+yOffForBounds)+h)

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
		if sys.argv[1] in ["preprocess", "generator_test", "crop_test", "box_test"]:
			mode = sys.argv[1]
	
	if mode == "unknown":
		print("mode not recognized")
	
	if mode == "preprocess":
		# remove all existing npy cache files
		allCaches = glob.glob("stage_1_train_images/*.npy")
		with closing(Pool(processes=multiprocessing.cpu_count()//2)) as pool:
		    pool.map(deleteFile, allCaches)
		    pool.terminate()
		# convert all png images to npy and save them
		allImages = glob.glob("stage_1_train_images/*.png")
		with closing(Pool(processes=multiprocessing.cpu_count()//2)) as pool:
		    pool.map(preprocessImage, allImages)
		    pool.terminate()
	
	if mode == "generator_test":
		generator = DCMGenerator(False, None, False)
	
		input,output,patientIds = generator.generateImages(1, 1.0)
		
		print("patientIds", patientIds)
		print("input", input)
		print("output", output)
	
	if mode == "crop_test":
		# show two images
		# one image is the normal size training image with boxes on it
		# second image is training size image with boxes on it, the boxes calculated from the output
		
		patientID = "49b95513-daab-49bb-bc6e-c5254ab1bc07"
		
		# ------------- First Image -----------------
		allPatients = GetPhase3PatientInfo()
		patient = None
		for otherPatient in allPatients:
			if otherPatient[kPatientID] == patientID:
				patient = otherPatient
		
		boxes = []
		for otherPatient in allPatients:
			if patient[kPatientID] == otherPatient[kPatientID]:
				boxes.append(minMaxBoxForPatient(otherPatient))
				
		# we have a patient, load the image
		dcmFilePath = dcmFilePathForTrainingPatient(patient)
		dcmData = pydicom.read_file(dcmFilePath)
		dcmImage = dcmData.pixel_array.astype('float32') / 255
		
		# render the boxes on the normal image and show it
		colorDcmImage = Image.fromarray(dcmImage * 255.0).convert("RGB")
		draw = ImageDraw.Draw(colorDcmImage)
		for box in boxes:
			draw.rectangle(box, outline="yellow")
		
		colorDcmImage.show()
		
		
		# ------------- Second Image -----------------
		generator = DCMGenerator(False, None, False)
		input,output = generator.generateImagesForPatient(patientID)
	
		for i in range(0,len(output)):
			sourceImg = Image.fromarray(input[i].reshape(IMG_SIZE[0],IMG_SIZE[1]) * 255.0).convert("RGB")
		
			draw = ImageDraw.Draw(sourceImg)
		
			boxes = generator.coordinatesFromOutput(output[i],IMG_SIZE)
			for box in boxes:
				draw.rectangle(box, outline="yellow")
			
			sourceImg.show()
		
	
	if mode == "box_test":
		# show two images:
		# one at normal size with the training boxes on it
		# another at cropped size with the training boxes on it
		allPatients = GetPhase3PatientInfo()
		
		# find a random patient with pneumonia
		patient = None
		while(True):
			randomPatient = random.choice(allPatients)
			if randomPatient[kTarget] == "1":
				patient = randomPatient
				break
				
		# find all of the pneumonia boxes for this patient
		boxes = []
		for otherPatient in allPatients:
			if patient[kPatientID] == otherPatient[kPatientID]:
				boxes.append(minMaxBoxForPatient(otherPatient))
		
		# we have a patient, load the image
		dcmFilePath = dcmFilePathForTrainingPatient(patient)
		dcmData = pydicom.read_file(dcmFilePath)
		dcmImage = dcmData.pixel_array.astype('float32') / 255
		
		# render the boxes on the normal image and show it
		colorDcmImage = Image.fromarray(dcmImage * 255.0).convert("RGB")
		draw = ImageDraw.Draw(colorDcmImage)
		for box in boxes:
			draw.rectangle(box, outline="yellow")
		
		colorDcmImage.show()
		
		# create the cropped image
		cropBox = [patient[kCropX],
			patient[kCropY],
			patient[kCropX] + patient[kCropWidth],
			patient[kCropY] + patient[kCropHeight]
		]
		croppedImage = dcmImage[int(cropBox[1]):int(cropBox[3]),int(cropBox[0]):int(cropBox[2])]
		
		# find all of the cropped pneumonia boxes for this patient
		boxes = []
		for otherPatient in allPatients:
			if patient[kPatientID] == otherPatient[kPatientID]:
				boxes.append(minMaxCroppedBoxForPatient(otherPatient, 0, 0))
		
		colorCroppedDcmImage = Image.fromarray(croppedImage * 255.0).convert("RGB")
		draw = ImageDraw.Draw(colorCroppedDcmImage)
		for box in boxes:
			draw.rectangle(box, outline="yellow")
		
		colorCroppedDcmImage.show()
		
		
		
	''''	
	#generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")
	generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")	
	generator.ignoreCaches = True
	
	input,output,patientIds = generator.generateImages(2,True,0.5)
	
	for i in range(0,len(input)):
				
		sourceImg = Image.fromarray(input[i].reshape(IMG_SIZE[0],IMG_SIZE[1]) * 255.0).convert("RGB")
		
		draw = ImageDraw.Draw(sourceImg)
		
		boxes = generator.coordinatesFromOutput(output[i],IMG_SIZE)
		for box in boxes:
			draw.rectangle(box, outline="white")
		
		sourceImg.save('/tmp/scan_%d_%s.png' % (i, generator.convertOutputToString(output[i])))
	'''
	
	