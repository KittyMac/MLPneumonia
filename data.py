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
		with open(labelsFile) as csv_file:
			self.labelsInfo = list(csv.reader(csv_file))
			self.labelsInfo.pop(0)
				
	
	def loadImageForPatientId(self,patientId):
		imageData = None
		
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
			image = image.resize((IMG_SIZE[0],IMG_SIZE[1]), Image.ANTIALIAS)
			image = image.convert('L')
			imageData = np.array(image).astype('float32').reshape(IMG_SIZE[0],IMG_SIZE[1],IMG_SIZE[2]) / 255
			
			print("caching image: %s" % cachedFilePath)
			np.save(cachedFilePath, imageData)
		
		if imageData is None and os.path.isfile(cachedFilePath) == True:
			imageData = np.load(cachedFilePath)
				
		return imageData
		
	
	def generateImages(self,num):
		randomSelection = True
		if num <= 0:
			num = len(self.labelsInfo)
			randomSelection = False
		
		input = np.zeros((num,IMG_SIZE[1],IMG_SIZE[0],IMG_SIZE[2]), dtype='float32')
		output = np.zeros((num,1), dtype='float32')
		
		for i in range(0,num):
			patient = random.choice(self.labelsInfo) if randomSelection else self.labelsInfo[i]
			imageData = self.loadImageForPatientId(patient[kPatientID])
			np.copyto(input[i], imageData)
			output[i][0] = patient[kTarget]
				
		return input,output
	
	def convertOutputToString(self,output):
		return np.array2string(output.astype(int), separator='.')
	
			

if __name__ == '__main__':
		
	#generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")
	generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_images.csv")	
	generator.ignoreCaches = True
	
	input,output = generator.generateImages(5)
	
	for i in range(0,len(input)):
		sourceImg = Image.fromarray(input[i].reshape(IMG_SIZE[0],IMG_SIZE[1]) * 255.0).convert("RGB")
				
		#draw = ImageDraw.Draw(sourceImg)
		#draw.rectangle(generator.GetCoordsFromOutput(output[n],size), outline="green")		
		
		sourceImg.save('/tmp/scan_%d_%s.png' % (i, generator.convertOutputToString(output[i])))
	
	