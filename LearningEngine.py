from random import randint
import numpy as np

class LearningEngine:
	def __init__(self, preknownRanks, preknownAnomalousNodes, dataSize):
		self.preknownRanks = preknownRanks
		self.preknownAnomalousNodes = preknownAnomalousNodes
		self.dataSize = dataSize
	
	#Assuming ranks is an array of nodes. Their position is defined by array index.
	def lossFunction(self, ranks):
		sumOfDistances = 0
		for i, rank in enumerate(ranks):
			j = self.dataSize
			if rank in preknownRanks:
				j = preknownRanks.index(rank)
			sumOfDistances += (i - j)^2
		return 1 - sumOfDistances / (self.dataSize * (self.dataSize ^ 2 - 1))

	def gradientDescent(self):
		randomSigma = random.uniform(0, 100)
		graphFactory = GraphFactory(matrix, randomSigma, 4, '/Users/Pragya/Documents/SDL/SLD-C1/SynesizedData/')
		graphFactory.createGraph()
		graphFactory2 = GraphFactory(newmatrix, randomSigma, 4, '/Users/Pragya/Documents/SDL/SLD-C1/SynesizedDataWithAnomalies/')
		graphFactory2.createGraph()
		# Run CAD
		oldSigma = randomSigma
		if self.preknownAnomalousNodes != anomalousNodes:
			newSigma = oldSigma - learningRate * lossFunction(ranksFromCAD)

