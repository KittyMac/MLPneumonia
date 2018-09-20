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

kMaxImageOffset = 10

class DCMGenerator():
	
	def __init__(self,finalSubmission):
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
		
		localLabelsInfo = self.labelsInfo[:]
		
		randomSelection = True
		if num <= 0:
			num = len(localLabelsInfo)
			randomSelection = False
		
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((num,IMG_SUBDIVIDE+IMG_SUBDIVIDE), dtype='float32')
		
		random.shuffle(localLabelsInfo)
		
		patientIds = []
		
		numPositive = 0
		numNegative = 0
		
		for i in range(0,num):
			
			if randomSelection:
				attempts = 10000
				while attempts > 0:
					patient = random.choice(localLabelsInfo)
					
					if numPositive <= (numNegative+numPositive)*positiveSplit and patient[kTarget] == "1":
						numPositive += 1
						break
					if numPositive > (numNegative+numPositive)*positiveSplit and patient[kTarget] == "0":
						numNegative += 1
						break
					
					attempts -= 1
			else:
				patient = localLabelsInfo[i]
			
			if num < len(localLabelsInfo):
				localLabelsInfo.remove(patient)
			
			patientIds.append(patient[kPatientID])
			
			input2,output2 = self.generateImagesForPatient(patient[kPatientID])
			
			np.copyto(input[i],input2[0])
			np.copyto(output[i],output2[0])
		
		if randomSelection:
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

		imageData = self.loadImageForPatientId(localPatient)
		np.copyto(input[0], imageData)
	
		if localPatient[kTarget] == "1":
			# note: we may have multiple data lines per patient, so we want to
			# combine their outputs such that there is only one combined training sample
			for patient in localPatientInfo:

				# this is the box at cropped image size & location
				box = minMaxCroppedBoxForPatient(patient)
				
				imgWidth = float(patient[kCropWidth])
				imgHeight = float(patient[kCropHeight])
		
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
				if output[x] >= 0.5 and output[IMG_SUBDIVIDE+y] >= 0.5:
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
			if output[x] >= 0.5:
				# ensure this peak is large enough (ignore little peaks)
				isPeak = False
				for x2 in range(x,IMG_SUBDIVIDE):
					if output[x2] < 0.5:
						if x2 - x > minPeakSize:
							isPeak = True
				
				# peak identified, fill it out
				if isPeak:
					peakIdx += 1
					for x2 in range(x,IMG_SUBDIVIDE):
						if output[x2] >= 0.5:
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
						if output[x] >= 0.5 and output[IMG_SUBDIVIDE+y] >= 0.5:
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
	
def minMaxCroppedBoxForPatient(patient):
	cx = float(patient[kCropX])
	cy = float(patient[kCropY])
	x = float(patient[kBoundsX])
	y = float(patient[kBoundsY])
	w = float(patient[kBoundsWidth])
	h = float(patient[kBoundsHeight])
	return ( x-cx, y-cy, (x-cx)+w, (y-cy)+h)

def preprocessImage(imageFile):
	imageData = cv2.imread(imageFile, 0)
	imageData = imageData.astype('float32') / 255
	imageData = cv2.resize(imageData, (IMG_SIZE[0],IMG_SIZE[1])).reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2])
	outputPath = imageFile[:-4]
	print("caching image: %s" % outputPath)
	np.save(outputPath, imageData)

if __name__ == '__main__':
	
	mode = "unknown"
	
	if len(sys.argv) >= 2:
		if sys.argv[1] in ["preprocess", "generator_test", "box_test"]:
			mode = sys.argv[1]
	
	if mode == "unknown":
		print("mode not recognized")
	
	if mode == "preprocess":
		# convert all png images to npy and save them
		allFiles = glob.glob("stage_1_train_images/*.png")
		for path in allFiles:
			preprocessImage(path)
	
	if mode == "generator_test":
		# show two images
		# one image is the normal size training image with boxes on it
		# second image is training size image with boxes on it, the boxes calculated from the output
		
		patientID = "34858b4b-37ff-4130-be8f-7075f3f3b056"
		
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
		generator = DCMGenerator(False)
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
				boxes.append(minMaxCroppedBoxForPatient(otherPatient))
		
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
	
	