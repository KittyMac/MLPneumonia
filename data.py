# script responsible for generating audio sample of specific sounds waves.  Supports
# exporting those samples to WAV files for confirmation.

from __future__ import division

import csv
import pydicom
import numpy as np
import os
import random
import os.path
from model import IMG_SIZE
from model import IMG_SUBDIVIDE
from PIL import Image,ImageDraw

kPatientID = 0
kBoundsX = 1
kBoundsY = 2
kBoundsWidth = 3
kBoundsHeight = 4
kTarget = 5

class DCMGenerator():
	
	def __init__(self,directory,labelsFile):
		self.directory = directory
		self.ignoreCaches = False
		self.labelsFile = labelsFile
		self.labelsInfo = []
		
		# load in all of the label info if it exists
		if labelsFile is not None:
			with open(labelsFile) as csv_file:
				self.labelsInfo = list(csv.reader(csv_file))
				self.labelsInfo.pop(0)
				
	
	def loadImageForPatientId(self,patient,withBox=False):
		imageData = None
		
		patientId = patient[kPatientID]
		
		# first check if a cached numpy array file already exists
		cachedFilePath = self.directory + "/" + patientId + ".npy"
		dcmFilePath = self.directory + "/" + patientId + ".dcm"
		
		if self.ignoreCaches == True or os.path.isfile(cachedFilePath) == False:
			# load the DCM and process it, saving the resulting numpy array to file
			dcmData = pydicom.read_file(dcmFilePath)
			imageData = dcmData.pixel_array
			
			# preprocess the image (reshape and normalize)
			# 1. resize it to IMG_SIZE (ie downsample)
			# 2. convert to float32
			# 3. reshape to greyscale
			# 4. normalize
			image = Image.fromarray(imageData).convert("RGB")
			
			if withBox and patient[kTarget] == "1":
				draw = ImageDraw.Draw(image)
				draw.rectangle(self.coordinatesFromPatient(patient), outline="white")		
			
			image = image.resize((IMG_SIZE[0],IMG_SIZE[1]), Image.ANTIALIAS)
			image = image.convert('L')
			imageData = np.array(image).astype('float32').reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]) / 255
			
			print("caching image: %s" % cachedFilePath)
			np.save(cachedFilePath, imageData)
		
		if imageData is None and os.path.isfile(cachedFilePath) == True:
			imageData = np.load(cachedFilePath)
				
		return imageData
		
	
	def generateImages(self,num,withBox=False):
		randomSelection = True
		if num <= 0:
			num = len(self.labelsInfo)
			randomSelection = False
		
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((num,IMG_SUBDIVIDE+IMG_SUBDIVIDE), dtype='float32')
		
		random.shuffle(self.labelsInfo)
		
		patientIds = []
		
		for i in range(0,num):
			patient = random.choice(self.labelsInfo) if randomSelection else self.labelsInfo[i]
			patientIds.append(patient[kPatientID])
			
			imageData = self.loadImageForPatientId(patient,withBox)
			np.copyto(input[i], imageData)

			if patient[kTarget] == "1":
				xmin = float(patient[kBoundsX])
				ymin = float(patient[kBoundsY])
				xmax = xmin + float(patient[kBoundsWidth])
				ymax = ymin + float(patient[kBoundsHeight])
				
				# Note: the canvas the bounds are in is 1024x1024
				xdelta = (1024 / IMG_SUBDIVIDE)
				ydelta = (1024 / IMG_SUBDIVIDE)
				for x in range(0, IMG_SUBDIVIDE):
					for y in range(0, IMG_SUBDIVIDE):
						xValue = x * xdelta
						yValue = y * ydelta
						if xValue+xdelta >= xmin and xValue <= xmax:
							output[i][x] = 1
						if yValue+ydelta >= ymin and yValue <= ymax:
							output[i][IMG_SUBDIVIDE+y] = 1
							
		return input,output,patientIds
	
	def generateImagesForPatient(self,patientID):
		
		num = 0
		for i in range(0,len(self.labelsInfo)):
			patient = self.labelsInfo[i]
			if patient[kPatientID] == patientID:
				num += 1
		
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((num,IMG_SUBDIVIDE+IMG_SUBDIVIDE), dtype='float32')
		
		idx = 0
		for i in range(0,len(self.labelsInfo)):
			patient = self.labelsInfo[i]
			if patient[kPatientID] == patientID:
				imageData = self.loadImageForPatientId(patient,False)
				np.copyto(input[idx], imageData)
			
				if patient[kTarget] == "1":
					xmin = float(patient[kBoundsX])
					ymin = float(patient[kBoundsY])
					xmax = xmin + float(patient[kBoundsWidth])
					ymax = ymin + float(patient[kBoundsHeight])
				
					# Note: the canvas the bounds are in is 1024x1024
					xdelta = (1024 / IMG_SUBDIVIDE)
					ydelta = (1024 / IMG_SUBDIVIDE)
					for x in range(0, IMG_SUBDIVIDE):
						for y in range(0, IMG_SUBDIVIDE):
							xValue = x * xdelta
							yValue = y * ydelta
							if xValue+xdelta >= xmin and xValue <= xmax:
								output[idx][x] = 1
							if yValue+ydelta >= ymin and yValue <= ymax:
								output[idx][IMG_SUBDIVIDE+y] = 1
				
				idx += 1
							
		return input,output
	
	def generatePredictionImages(self):
		
		fileList = []
		for file in os.listdir(self.directory):
		    if file.endswith(".dcm"):
				fileList.append(os.path.splitext(file)[0])
		
		num = len(fileList)
				
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
				
		for i in range(0,num):
			patient = [fileList[i]]
			imageData = self.loadImageForPatientId(patient,False)
			np.copyto(input[i], imageData)
										
		return fileList,input
	
	def convertOutputToString(self,output):
		return 1 if np.max(output) >= 0.5 else 0
		#return np.array2string(output.astype(int), separator='.')
	
	def coordinatesFromPatient(self,patient):
		x = float(patient[kBoundsX])
		y = float(patient[kBoundsY])
		w = float(patient[kBoundsWidth])
		h = float(patient[kBoundsHeight])
		return ((x,y),(x+w,y+h))
	
	def coordinatesFromOutput(self,output,size):
		IMG_SUBDIVIDE = int(len(output)/2)
		
		xdelta = 1.0 / IMG_SUBDIVIDE
		ydelta = 1.0 / IMG_SUBDIVIDE
	
		xmin = 1.0
		xmax = 0.0
		ymin = 1.0
		ymax = 0.0
	
		for x in range(0,IMG_SUBDIVIDE):
			for y in range(0,IMG_SUBDIVIDE):
				xValue = (x*xdelta)
				yValue = (y*ydelta)
				
				if output[x] >= 0.5 and output[IMG_SUBDIVIDE+y] >= 0.5:
					if xValue < xmin:
						xmin = xValue
					if xValue > xmax:
						xmax = xValue
					if yValue < ymin:
						ymin = yValue
					if yValue > ymax:
						ymax = yValue
				
				#if output[x] >= 0.5:
				#	if xValue < xmin:
				#		xmin = xValue
				#	if xValue > xmax:
				#		xmax = xValue
				#if output[IMG_SUBDIVIDE+y] >= 0.5:
				#	if yValue < ymin:
				#		ymin = yValue
				#	if yValue > ymax:
				#		ymax = yValue

		return (xmin*size[1],ymin*size[0],xmax*size[1],ymax*size[0])
	
			

if __name__ == '__main__':
		
	#generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")
	generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")	
	generator.ignoreCaches = True
	
	input,output,patientIds = generator.generateImages(20, True)
	
	for i in range(0,len(input)):
		sourceImg = Image.fromarray(input[i].reshape(IMG_SIZE[0],IMG_SIZE[1]) * 255.0).convert("RGB")
		
		draw = ImageDraw.Draw(sourceImg)
		draw.rectangle(generator.coordinatesFromOutput(output[i],IMG_SIZE), outline="white")
		
		sourceImg.save('/tmp/scan_%d_%s.png' % (i, generator.convertOutputToString(output[i])))
	
	