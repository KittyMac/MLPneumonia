from __future__ import division

import signal
import numpy as np
import random
import time
import bisect
import multiprocessing
from multiprocessing import Process
import sys
import os

class GracefulKiller:
	kill_now = False
	def __init__(self):
		signal.signal(signal.SIGINT, self.exit_gracefully)
		signal.signal(signal.SIGTERM, self.exit_gracefully)
		signal.signal(signal.SIGUSR1, self.exit_gracefully)

	def exit_gracefully(self,signum, frame):
		self.kill_now = True

class GeneticAlgorithm:
	
	prng = random
	
	# population size
	numberOfOrganisms = 20
	
	# generate organisms: delegate received the population index of the new organism, and a uint suitable for seeding a RNG. delegete should return a newly allocated organism with assigned chromosomes.
	generateOrganism = None
	
	# reset organisms: delegate receives the entire population, allowing them to seed the population with the most useful data possible to help speed up computation
	# this should get called when the population is initially generated.
	resetOrganisms = None
	
	# breed organisms: delegate is given two parents, a child, and a uint suitable for seeding a RNG. delegete should fill out the chromosomes of the child with chromosomes selected from each parent,
	# along with any possible mutations which might occur.
	breedOrganisms = None
	
	# score organism: delegate is given and organism and should return a float value representing the "fitness" of the organism. Higher scores must always be better scores!
	scoreOrganism = None
	
	# choose organism: delegate is given an organism, its fitness score, and the number of generations processed so far. return true to signify this organism's answer is
	# sufficient and the genetic algorithm should stop; return false to tell the genetic algorithm to keep processing.
	chosenOrganism = None
	
	# counter to keep track of total number of generations during perform genetics
	masterGenerations = 0
	
	
	# tweening method used by PerformGenetics() to aid in the selection of parents to breed; easeInExpo will favor parents with bad fitness values
	def easeInExpo (self, start, end, val):
		return (end - start) * 2**(10 * (val / 1 - 1)) + start
	
	# tweening method used by PerformGenetics() to aid in the selection of parents to breed; easeOutExpo will favor parents with good fitness values
	def easeOutExpo (self, start, end, val):
		return (end - start) * (-(2**(-10 * val / 1)) + 1) + start
	
	#@profile
	def _PerformGenetics (self, millisecondsToProcess, patience, generateOrganism, resetOrganisms, breedOrganisms, scoreOrganism, chosenOrganism, sharedOrganismIdx=-1, neighborOrganismIdx=-1):
		
		killer = GracefulKiller()
		
		localPRNG = self.prng
		
		# Replacement window is the number of organisms in the population we should "shuffle down" when we insert a newly born child into the population
		# This is generally an optimization step for large populations, as shifting the entire array each time a new child is inserted is expensive (and
		# generally not necessary).
		localNumberOfOrganisms = self.numberOfOrganisms
		localNumberOfOrganismsMinusOne = localNumberOfOrganisms - 1
		replacementWindow = 2
		
		# simple counter to keep track of the number of generations (parents selected to breed a child) have passed
		numberOfGenerations = 0
		patienceCount = 0
		
		# Create the population arrays; one for the organism classes and another to hold the scores of said organisms
		allOrganisms = [None]*localNumberOfOrganisms
		allOrganismScores = np.zeros((localNumberOfOrganisms), dtype=float)
		
		# Generate all of the organisms in the population array; score them as well
		for i in range(0,localNumberOfOrganisms):
			allOrganisms [i] = generateOrganism (i, localPRNG)
		
		if resetOrganisms is not None:
			resetOrganisms(allOrganisms, localPRNG)
		
		for i in range(0,localNumberOfOrganisms):
			allOrganismScores [i] = scoreOrganism (allOrganisms [i], sharedOrganismIdx, localPRNG)
		
		# sort the organisms so the higher fitness are all the end of the array; it is critical
		# for performance that this array remains sorted during processing (it eliminates the need
		# to search the population for the best organism).		
		sortedIndex = np.argsort(allOrganismScores)
		allOrganismScores = allOrganismScores[sortedIndex]
		allOrganisms = [allOrganisms[i] for i in sortedIndex]
				
		# timer to let us know when we've exceeded our alloted time
		watchStart = time.time()
		
		# create a new "child" organism. this is an optimization, in order to remove the need to allocate new children
		# during breeding, as designate one extra organsism as the "child".  We then shuffle this in and out of the
		# population array when required, eliminating the need for costly object allocations
		newChild = generateOrganism (0, localPRNG)
		trashedChild = newChild
		childScore = scoreOrganism (newChild, sharedOrganismIdx, localPRNG)
		
		# The multi-threaded version of this relies on a ring network of threads to process; if this is the multithreaded version
		# then we need to include an extra generation step during processing (see comments PerformGeneticsThreaded for overview)
		maxBreedingPerGeneration = 3
		if (neighborOrganismIdx >= 0):
			maxBreedingPerGeneration = 4		
		
		# used in parent selection for breeders below
		a = 0.0
		b = 0.0
		didFindNewBestOrganism = False
		
		# Check to see if we happen to already have the answer in the starting population
		if (chosenOrganism (allOrganisms [localNumberOfOrganismsMinusOne], allOrganismScores [localNumberOfOrganismsMinusOne], numberOfGenerations, sharedOrganismIdx, localPRNG) == False):
			
			localEaseOutExpo = self.easeOutExpo
			localEaseInExpo = self.easeInExpo
			
			while (killer.kill_now == False and ((time.time() - watchStart) * 1000) < millisecondsToProcess and patienceCount < patience):
				
				# optimization: we only call chosen organsism below when the new best organism changes
				didFindNewBestOrganism = False
								
				# we use three (or four) methods of parent selection for breeding; this iterates over all of those
				for i in range(0,maxBreedingPerGeneration):
					
					# Below we have four different methods for selecting parents to breed. Each are explained individually
					if (i == 0):
						# Breed the pretty ones together: favor choosing two parents with good fitness values
						a = int(localEaseOutExpo (0.25, 1.0, localPRNG.random()) * localNumberOfOrganisms)
						b = int(localEaseOutExpo (0.25, 1.0, localPRNG.random()) * localNumberOfOrganisms)
						breedOrganisms (allOrganisms [a], allOrganisms [b], newChild, localPRNG)
					elif (i == 1):
						# Breed a pretty one and an ugly one: favor one parent with a good fitness value, and another parent with a bad fitness value
						a = int(localEaseInExpo (0.0, 0.5, localPRNG.random()) * localNumberOfOrganisms)
						b = int(localEaseOutExpo (0.5, 1.0, localPRNG.random()) * localNumberOfOrganisms)
						breedOrganisms (allOrganisms [a], allOrganisms [b], newChild, localPRNG)
					elif (i == 2):
						# Breed the best organism asexually: IT IS BEST IF THE BREEDORGANISM DELEGATE CAN RECOGNIZE THIS AND FORCE A HIGHER RATE OF SINGLE CHROMOSOME MUTATION
						breedOrganisms (allOrganisms [localNumberOfOrganismsMinusOne], allOrganisms [localNumberOfOrganismsMinusOne], newChild, localPRNG)
					elif (i == 3):
						# Breed the best organism of my neighboring thread in the ring network asexually into our population
						if numberOfGenerations % (sharedOrganismIdx + 1000) == 0:
							try:
								neighborOrganism = self.sharedOrganismQueue.get_nowait()
								if (neighborOrganism == None):
									breedOrganisms (allOrganisms [localNumberOfOrganismsMinusOne], allOrganisms [localNumberOfOrganismsMinusOne], newChild, localPRNG)
								#print("neighborOrganism: {}".format(neighborOrganism))
								breedOrganisms (neighborOrganism, neighborOrganism, newChild, localPRNG)
							except:
								breedOrganisms (allOrganisms [localNumberOfOrganismsMinusOne], allOrganisms [localNumberOfOrganismsMinusOne], newChild, localPRNG)
						else:
							breedOrganisms (allOrganisms [localNumberOfOrganismsMinusOne], allOrganisms [localNumberOfOrganismsMinusOne], newChild, localPRNG)
		
				
					# record the fitness value of the newly bred child
					childScore = scoreOrganism (newChild, sharedOrganismIdx, localPRNG)

					# if we're better than the worst member of the population, then the child should be inserted into the population
					if (childScore > allOrganismScores [0]):
						
						left = bisect.bisect(allOrganismScores, childScore)-1
										
						# sanity check: ensure we've got a better score than the organism we are replacing
						if (childScore > allOrganismScores [left]):
								
							# when we insert a new child into the population, we "shuffle down" existing organisms to make
							# room for the new guy, allowing us to euthenize a worse organism while keeping the
							# strong organisms. as an optimization, we don't do the entire population array, instead
							# we do replacementWindow number of organisms
							startReplacementWindow = (left - replacementWindow if left - replacementWindow >= 0 else 0)
	
							# note: we need to juggle the organism we're going to trash, as it will become our
							# newChild replacement object ( so we recycle organisms instead of creating new ones )
							trashedChild = allOrganisms [startReplacementWindow]
	
							# "shuffle down" our replacement window
							if((startReplacementWindow + 1) <= left):
								for j in range(startReplacementWindow+1,left+1):
									allOrganismScores [j - 1] = allOrganismScores [j];
									allOrganisms [j - 1] = allOrganisms [j];
	
							# insert the new child into the population
							allOrganisms [left] = newChild;
							allOrganismScores [left] = childScore;
	
							# reuse our trashed organism
							newChild = trashedChild;
	
							# if we have discovered a new best organism
							if (left == localNumberOfOrganismsMinusOne):
								# set this flag to ensure chosenOrganism() gets called
								didFindNewBestOrganism = True;
				
				
				# update the number of generations we have now processed
				numberOfGenerations += maxBreedingPerGeneration
				
				# reset the patience counter when we find new best score
				if didFindNewBestOrganism:
					patienceCount = 0
				
				# reset the population as we approach our patience goal to allow for
				# climbing out of local minimums
				for i in range(0, maxBreedingPerGeneration):
					patienceCount += 1
					if patienceCount == (patience // 4) or \
						patienceCount == (patience // 3) or \
						patienceCount == (patience // 2) or \
						patienceCount == (patience * 3) // 4:
						if resetOrganisms is not None:
							
							oldBestScore = allOrganismScores [localNumberOfOrganismsMinusOne]
							
							# when we reset the population, we must re-score and re-sort the adjusted population as well
							numberOfOrganismsReset = resetOrganisms(allOrganisms, localPRNG)

							for i in range(0,numberOfOrganismsReset):
								allOrganismScores [i] = scoreOrganism (allOrganisms [i], sharedOrganismIdx, localPRNG)

							sortedIndex = np.argsort(allOrganismScores)
							allOrganismScores = allOrganismScores[sortedIndex]
							allOrganisms = [allOrganisms[i] for i in sortedIndex]
							
							newBestScore = allOrganismScores [localNumberOfOrganismsMinusOne]
							
							print("reset %d organisms, patience is %d, best score was %f and now is %f" % (numberOfOrganismsReset, patienceCount, oldBestScore, newBestScore))
				
				# if we found a new best organism we should share it
				if didFindNewBestOrganism and sharedOrganismIdx >= 0 and random.random() < 0.1:
					self.sharedOrganismQueue.put_nowait(allOrganisms [localNumberOfOrganismsMinusOne])

				# if we found a new best organism, check with our delegate to see if we need to continue processing or not
				if (didFindNewBestOrganism and chosenOrganism (allOrganisms [localNumberOfOrganismsMinusOne], allOrganismScores [localNumberOfOrganismsMinusOne], numberOfGenerations, sharedOrganismIdx, localPRNG)):
					# if we're multi-threaded and we found the correct answer, make sure to let all of the other ring-threads know so they can stop too
					if (sharedOrganismIdx >= 0):
						self.sharedOrganismsDone = True
					break
				
				
		# note: for proper reporing of number of generations processed, we need to call choose organism one more time before exiting
		chosenOrganism (allOrganisms [localNumberOfOrganismsMinusOne], allOrganismScores [localNumberOfOrganismsMinusOne], numberOfGenerations, sharedOrganismIdx, localPRNG)
	
		return allOrganisms [localNumberOfOrganismsMinusOne],allOrganismScores [localNumberOfOrganismsMinusOne]
	
	
	
	def _CaptureMasterGenerationChosenPassthrough(self, organism, score, generation, sharedOrganismIdx, prng):
		self.masterGenerations = generation
		return self.chosenOrganism(organism, score, generation, sharedOrganismIdx, prng)
	
	def PerformGenetics (self, millisecondsToProcess, patience):
		
		watchStart = time.time()
		self.masterGenerations = 0

		bestOrganism,bestOrganismScore = self._PerformGenetics (millisecondsToProcess, patience, self.generateOrganism, self.resetOrganisms, self.breedOrganisms, self.scoreOrganism, self._CaptureMasterGenerationChosenPassthrough)
	
		watchEnd = time.time()
	
		print("Done in {}ms and {} generations".format(int((watchEnd-watchStart)*1000), self.masterGenerations))
	
		return (bestOrganism,bestOrganismScore)
	
	
	
	# Perform the genetic algorithm on many threads (as many threads as we have processing cores, specifically).
	# To do this, we take on a ring network paradigm for the threads.  Each thread is created and given its
	# own population of organisms which it processes separately from all other threads. Threads then
	# pass the best organism in their population to their neighboring thread (one way only, and the neighbors
	# wrap so that there is a full ring of communication). During the parent selection phases during breeding
	# one of the selection methods is to asexually breed this shared organism into the thread's population.
	# Using this scheme, each thread is able to process in a highly parallizable fashion while still incorporating
	# the best chromosomes of other threads into its population
	#
	# Python notes: Since python doesn't have true multithreading we're using multiprocessing instead. This means
	# to facilite the ring network we cannot use shared memory (aka sharedOrganisms array) and instead need to
	# set up pipes between all neighbors.  The nodes share between themselves until they are complete, at which
	# point they pass it up to the parent process using a queue (fifo).  The parent aggregates the results
	# and chooses the best one as the actual result.
	sharedOrganismsDone = False
	
	# when each thread ends, it will check to see if its chosen plan is better than the master plan and replace it,
	# allowing us to return the best plan conceived over all of the threads in the ring network
	masterBestOrganism = None
	masterBestOrganismScore = 0
	masterGenerations = 0
	numberOfRunningThreads = 0
	
	# communication to and from parent / neighbor processes
	workerToParentQueue = None
	sharedOrganismQueue = None
		
	def PerformGeneticsThreaded (self, millisecondsToProcess, patience):
		numThreads = multiprocessing.cpu_count()
		if numThreads == 1:
			return PerformGenetics (millisecondsToProcess, patience)
			
		watchStart = time.time()
		
		self.masterBestOrganism = None
		self.masterBestOrganismScore = -999999999999.0
		self.masterGenerations = 0
		self.numberOfRunningThreads = 0
		
		# allocate our shared oragnisms array to allow threads to pass their best organisms along
		self.sharedOrganisms = [None]*numThreads
		self.sharedOrganismsDone = False
		
		manager = multiprocessing.Manager()
		
		self.workerToParentQueue = manager.Queue()
		self.sharedOrganismQueue = manager.Queue()
		
		allProcesses = []
		
		for i in range(0,numThreads):
			self.numberOfRunningThreads += 1
			p = Process(target=self._WorkerProcess, args=(random.random()*9999999,millisecondsToProcess,patience,i,numThreads,))
			p.start()
			allProcesses.append(p)
		
		# lock waiting for all threads to finish
		for i in range(0,numThreads):
			(bestOrganism,bestScore,totalGenerations,earlyFinish) = self.workerToParentQueue.get()
			self.masterGenerations += totalGenerations
			if bestScore > self.masterBestOrganismScore:
				self.masterBestOrganismScore = bestScore
				self.masterBestOrganism = bestOrganism
			
			# one of the workers found the full answer, signal all of the others to stop process and report their results
			if earlyFinish:
				for p in allProcesses:
					os.kill(p.pid, signal.SIGUSR1)
		
		

		self.sharedOrganismsDone = False

		watchEnd = time.time()
		print("Done in {}ms and {} generations".format(int((watchEnd-watchStart)*1000), self.masterGenerations))

		return self.masterBestOrganism
		
	def _WorkerProcess(self, randomSeed,millisecondsToProcess,patience,sharedOrganismIdx,numThreads):
		
		self.prng.seed(randomSeed)
		
		bestOrganism,bestOrganismScore = self._PerformGenetics (millisecondsToProcess, patience, self.generateOrganism, self.resetOrganisms, self.breedOrganisms, self.scoreOrganism, self._CaptureMasterGenerationChosenPassthrough, sharedOrganismIdx, (sharedOrganismIdx + 1) % numThreads)
		
		# pass our results back up to the parent process
		self.workerToParentQueue.put((bestOrganism,self.scoreOrganism (bestOrganism, sharedOrganismIdx, self.prng),self.masterGenerations, self.sharedOrganismsDone))
		
		