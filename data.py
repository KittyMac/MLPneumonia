# script responsible for generating audio sample of specific sounds waves.  Supports
# exporting those samples to WAV files for confirmation.

from __future__ import division

import csv
import pydicom
import numpy as np
import os
import json
import operator
import random
import time
import math
import wave

import signal
import time

import scipy.io.wavfile

kPatientID = 0
kBoundsX = 1
kBoundsY = 2
kBoundsWidth = 3
kBoundsHeight = 4
kTarget = 5

class DCMGenerator():
	
	def __init__(self,directory,labelsFile,forceImageProcessing):
		self.directory = directory
		self.labelsFile = labelsFile
		self.labelsInfo = []
		
		# load in all of the label info if it exists
		with open(labelsFile) as csv_file:
			self.labelsInfo = list(csv.reader(csv_file))
			self.labelsInfo.pop(0)
		
		for patient in self.labelsInfo:
			self.loadImageForPatientId(patient[kPatientID],forceImageProcessing)
		
	
	def loadImageForPatientId(self,patientId,forceImageProcessing):
		imageData = None
		
		# first check if a cached numpy array file already exists
		cachedFilePath = self.directory + "/" + patientId + ".npy"
		dcmFilePath = self.directory + "/" + patientId + ".dcm"
		try:
			if forceImageProcessing:
				raise ValueError('Forcing generation of cached image')
			imageData = np.load(cachedFilePath)
		except:
			# load the DCM and process it, saving the resulting numpy array to file
			dcmData = pydicom.read_file(dcmFilePath)
			imageData = dcmData.pixel_array
			
			# preprocess the image (reshape and normalize)
			imageData = imageData.astype('float32').reshape(1024,1024,1) / 255
			
			print("caching image: %s" % cachedFilePath)
			np.save(cachedFilePath, imageData)
		
		return imageData
		
	
	def generateImages(self,num):
		# grab a random number of images from the source directory
		input_sounds = np.zeros((num,int(self.sampleRate*self.duration)), dtype='float32')
		output_sounds = np.zeros((num,2), dtype='float32')
				
		return input_sounds,output_sounds
			

if __name__ == '__main__':
		
	generator = DCMGenerator("data/stage_1_train_images", "data/stage_1_train_labels.csv", True)
	#input_sounds,output_sounds = generator.generateImages(5)
	
	#for i in range(0,len(input_sounds)):
	#	generator.saveSoundToFile("/tmp/generated%d.wav" % (i), input_sounds[i])
	
	