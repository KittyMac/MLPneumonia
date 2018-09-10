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
		self.x = width / 2
		self.y = height / 2
		self.gridSize = 120
	
	def __str__(self):
		return "(%d,%d,%d,%d)" % (self.x, self.y, self.x+self.gridSize, self.y+self.gridSize)
	def __repr__(self):
		return "(%d,%d,%d,%d)" % (self.x, self.y, self.x+self.gridSize, self.y+self.gridSize)
	
	def randomize(self,index,prng):
		if index == 0:
			self.x = prng.randint(0,self.width-self.gridSize)
		else:
			self.y = prng.randint(0,self.height-self.gridSize)
	
	def randomizeOne(self,prng):
		self.randomize(prng.randint(0,3), prng)
		
	def randomizeAll(self,prng):
		self.x = prng.randint(0,self.width-self.gridSize)
		self.y = prng.randint(0,self.height-self.gridSize)
		
	def copyFrom(self,other):
		self.x = other.x
		self.y = other.y
	
	def box(self):
		return (int(self.x),int(self.y),int(self.x+self.gridSize),int(self.y+self.gridSize))
	
	def crop(self,npImage):
		return npImage[int(self.y):int(self.y+self.gridSize),int(self.x):int(self.x+self.gridSize)]
		

class GeneticLocalization:
	
	# the genetic algorithm provides the crop
	# opencv getPerspectiveTransform converts the crop to image sized for predicting against the model
	# the prediction against the model is used at the fitness score for the genetic algorithm

	def __init__(self,npImage,cnnModel,ignoreBoxes,cnnModelImageSize):
		self.npImage = npImage
		self.cnnModel = cnnModel
		self.cnnModelImageSize = cnnModelImageSize
		self.ignoreBoxes = ignoreBoxes
		
		self.ga = GeneticAlgorithm()
		self.ga.numberOfOrganisms = 512
		
		def generateOrganism (idx,prng):
			o = Organism (1024,1024)
			o.randomizeAll (prng)
			return o
		self.ga.generateOrganism = generateOrganism
		
		def breedOrganisms(organismA, organismB, child, prng):
			if (organismA == organismB):
				child.copyFrom(organismA)
				child.randomizeOne(prng)
			else:
				child.x = organismA.x if prng.random() > 0.5 else organismB.x
				child.y = organismA.y if prng.random() > 0.5 else organismB.y

				if prng.random() > 0.5:
					child.randomizeOne(prng)
				elif prng.random() > 0.5:
					child.randomizeAll(prng)
		self.ga.breedOrganisms = breedOrganisms
		
		def scoreOrganism (organism, idx, prng):
			# if we overlap any ignore boxes by a significant portion we should score low...
			organismBox = organism.box()
			for box in self.ignoreBoxes:
				if self.calculateIntersectionOverUnion(box, organismBox) > 0.2:
					return 0.0
			
			cropped = organism.crop(self.npImage)
			input = cv2.resize(cropped, self.cnnModelImageSize)
			output = self.cnnModel.predict(input.reshape(1,self.cnnModelImageSize[0],self.cnnModelImageSize[1], 1))
			return output[0][0]
		self.ga.scoreOrganism = scoreOrganism
		
		def chosenOrganism(organism, score, generation, sharedOrganismIdx, prng):
			if score >= 1.0:
				return True
			return False
		self.ga.chosenOrganism = chosenOrganism
	
	def calculateIntersectionOverUnion(self, boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
 
		# compute the area of intersection rectangle
		interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)
 
		# return the intersection over union value
		return iou
	

	def findBox(self):
		bestOrgansim, bestScore = self.ga.PerformGenetics(60000)
		if bestScore >= 0.99:
			print("bestScore", bestScore, bestOrgansim.box())
			return bestOrgansim.box()
		return None
		