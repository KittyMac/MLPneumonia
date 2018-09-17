from __future__ import division

import numpy as np
import random
from PIL import Image,ImageDraw
from GeneticAlgorithm import GeneticAlgorithm
import cv2

class Organism:
	# an organism is a box crop of the original image
	def __init__(self,width,height):
		self.width = width
		self.height = height
		self.xmin = 0
		self.xmax = width
		self.ymin = 0
		self.ymax = height
		
		self.minSize = 128
	
	def __str__(self):
		return "(%d,%d,%d,%d)" % (self.xmin, self.ymin, self.xmax, self.ymax)
	def __repr__(self):
		return "(%d,%d,%d,%d)" % (self.xmin, self.ymin, self.xmax, self.ymax)
	
	def randomize(self,index,prng):
		if index == 0:
			self.xmin = prng.randint(0,self.xmax-self.minSize)
		elif index == 1:
			self.ymin = prng.randint(0,self.ymax-self.minSize)
		elif index == 2:
			self.xmax = prng.randint(self.xmin+self.minSize,self.width)
		else:
			self.ymax = prng.randint(self.ymin+self.minSize,self.height)
	
	def randomizeOne(self,prng):
		self.randomize(prng.randint(0,3), prng)
		
	def randomizeAll(self,prng):
		self.xmin = prng.randint(0,self.xmax-self.minSize)
		self.ymin = prng.randint(0,self.ymax-self.minSize)
		self.xmax = prng.randint(self.xmin+self.minSize,self.width)
		self.ymax = prng.randint(self.ymin+self.minSize,self.height)
	
	def validate(self):
		if self.xmax < self.xmin:
			t = self.xmin
			self.xmin = self.xmax
			self.xmax = t
		if self.ymax < self.ymin:
			t = self.ymin
			self.ymin = self.ymax
			self.ymax = t
		
		if self.xmax == self.xmin:
			self.xmin = 0
			self.xmax = self.width
		if self.ymax == self.ymin:
			self.ymin = 0
			self.ymax = self.height
		
	
	def copyFrom(self,other):
		self.xmin = other.xmin
		self.xmax = other.xmax
		self.ymin = other.ymin
		self.ymax = other.ymax
	
	def box(self):
		return (int(self.xmin),int(self.ymin),int(self.xmax),int(self.ymax))
	
	def crop(self,npImage):
		return npImage[int(self.ymin):int(self.ymax),int(self.xmin):int(self.xmax)]
		

class GeneticLocalization:
	
	# the genetic algorithm provides the crop
	# opencv getPerspectiveTransform converts the crop to image sized for predicting against the model
	# the prediction against the model is used at the fitness score for the genetic algorithm

	def __init__(self,npImage,cnnModel,cnnModelImageSize):
		self.npImage = npImage
		self.cnnModel = cnnModel
		self.cnnModelImageSize = cnnModelImageSize
		
		self.ga = GeneticAlgorithm()
		self.ga.numberOfOrganisms = 512
		
		def generateOrganism (idx,prng):
			o = Organism (1024,1024)
			o.randomizeAll (prng)
			o.validate ()
			return o
		self.ga.generateOrganism = generateOrganism
		
		def breedOrganisms(organismA, organismB, child, prng):
			if (organismA == organismB):
				child.copyFrom(organismA)
				child.randomizeOne(prng)
			else:
				child.xmin = organismA.xmin if prng.random() > 0.5 else organismB.xmin
				child.xmax = organismA.xmax if prng.random() > 0.5 else organismB.xmax
				child.ymin = organismA.ymin if prng.random() > 0.5 else organismB.ymin
				child.ymax = organismA.ymax if prng.random() > 0.5 else organismB.ymax

				if prng.random() > 0.5:
					child.randomizeOne(prng)
				elif prng.random() > 0.5:
					child.randomizeAll(prng)
			child.validate()
		self.ga.breedOrganisms = breedOrganisms
		
		def scoreOrganism (organism, idx, prng):
			try:
				cropped = organism.crop(self.npImage)
				input = cv2.resize(cropped, self.cnnModelImageSize)
				output = self.cnnModel.predict(input.reshape(1,self.cnnModelImageSize[0],self.cnnModelImageSize[1], 1))
				return output[0][0]
			except:
				print(self.npImage.shape)
				print(organism.crop(self.npImage))
				print("exception when scoring organism: ", organism)
				return 0.0
		self.ga.scoreOrganism = scoreOrganism
		
		def chosenOrganism(organism, score, generation, sharedOrganismIdx, prng):
			if score >= 0.999:
				return True
			return False
		self.ga.chosenOrganism = chosenOrganism
		

	def findBox(self):
		bestOrgansim, bestScore = self.ga.PerformGenetics(60000, 5000)
		if bestScore >= 0.5:
			print("bestScore", bestScore, bestOrgansim.box())
			return bestOrgansim.box()
		return None
		
