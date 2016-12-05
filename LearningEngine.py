from sklearn.preprocessing import normalize
import random
import numpy as np, subprocess, json
from scipy.io import loadmat, savemat
from sets import Set
import matplotlib.pyplot as plt
from GraphFactory import *

class LearningEngine:
	def __init__(self, configFile):
		print configFile
		with open(configFile) as config_file:
			data = json.load(config_file)
			self.spark = data["spark"]
			self.CAD = data["CAD"]
			self.CADOptions = data["CADOptions"]
			self.nodeFolder1 = data["nodeFolder1"]
			self.nodeFolder2 = data["nodeFolder2"]
			self.dataSize = data["totalDataSize"]
			self.resultsFolder = data["resultsFolder"]
			self.learningRate = data["learningRate"]
		data = loadmat(self.resultsFolder + 'GTresult.mat')
		self.deltaEGT = data["deltaE"]
		self.nodesGT = data["nodes"]

	def runForDifferentRates(self, x, y, spacing, iterations):
		graphVarsForRates = {}
		for rate in self.learningRate:
			graphVars = self.runForOneRate(x, y, spacing, iterations, rate)
			graphVarsForRates[rate] = graphVars
	
	def runWithRestarts(self, x, y, spacing, iterations, learningRate, noOfRestarts):
			graphVarsForRestarts = {}
			for i in range(noOfRestarts):
				graphVars = self.runForOneRate(x, y, spacing, iterations, rate)
				graphVarsForRates[i] = graphVars

	def runForOneRate(self, x, y, spacing, iterations, learningRate):
		sigma = random.uniform(0, 100)
		# [iteration, sigma, loss, learningRate]
		graphVars = [[0, sigma, 0, learningRate]]
		for i in range(iterations):
			graphFactory = GraphFactory(x, sigma, spacing, self.nodeFolder1, True)
			graphFactory.createGraph()
			graphFactory2 = GraphFactory(y, sigma, spacing, self.nodeFolder2, True)
			graphFactory2.createGraph()
			self.runCAD()
			data = loadmat(self.resultsFolder + 'result.mat')
			deltaE = data["deltaE"]
			nodes = data["nodes"]
			# if len(nodes.symmetric_difference(self.nodesGT)) == 0:
			# 	print "Found sigma ", sigma
			# 	break
			loss, newSigma = self.gradientDescent(sigma, deltaE, learningRate)
			print "Loss function", loss
			print "New sigma", sigma
			graphVars.append([i+1, sigma, loss, learningRate])
			if sigma == newSigma:
				print "Sigma didn't change in the last iteration"
				break
			else:
				sigma = newSigma
		print "Graph elements ", graphVars
		plt.plot(graphVars[:][0],graphVars[:][1], 'o')
		plt.plot(graphVars[:][0],graphVars[:][2], '+')
		plt.show()
		return graphVars
	
	#Assuming ranks is an array of nodes. Their position is defined by array index.
	def lossFunctionKamalika(self, ranks, preknownRanks):
		sumOfDistances = 0
		for i, rank in enumerate(ranks):
			j = self.dataSize
			if rank in preknownRanks:
				j = preknownRanks.index(rank)
			sumOfDistances += (i - j)^2
		return 1 - sumOfDistances / (self.dataSize * (self.dataSize ^ 2 - 1))

	def gradientDescent(self, sigma, deltaE, learningRate):
		sum = self.lossFunction(deltaE)
		return sum, sigma - learningRate * sum

	def runCAD(self):
		print "Starting CAD"
		cmd = [self.spark, self.CAD, self.CADOptions]
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
		out, err = p.communicate()
		print out

	def convertToInteger(self, deltaE):
		deltaE[:, 0] = np.asarray(deltaE[:, 0]).astype(int)
		deltaE[:, 1] = np.asarray(deltaE[:, 1]).astype(int)
		return deltaE

	def lossFunction(self, deltaE):
		# print("======================== edge list of deltaE =========================")
		delta_edge = self.convertToInteger(deltaE[:,:2])
		print(delta_edge)
		# print("======================== edge list of groundTruth deltaE =============")
		groundtruth_edge = self.convertToInteger(self.deltaEGT[:,:2])
		print(groundtruth_edge)
		sum = 0
		# print("=======================================================================")
		# Iterate edge in groundtruthdEdge. If there is same edge in delteEdge, sum+= (score1 - score2)**2; otherwise, sum+=score1**2
		for i in range(len(groundtruth_edge)):
			sum += (self.deltaEGT[i][2])**2
			for j in range(len(delta_edge)):
				# print "deltaEdge",i,": ",delta_edge[j]
				# print "deltaGroundTruth", j, ": ",groundtruth_edge[i]
				if (delta_edge[j] == groundtruth_edge[i]).all() or (delta_edge[j] == groundtruth_edge[i][::-1]).all() :
					# print "true"
					sum += (self.deltaEGT[i][2] - deltaE[j][2])**2
					sum -= (self.deltaEGT[i][2])**2
					# print "sum is", sum
					break		
		# Iterate edge in delteEdge. If there no same edge in groundtruthdEdge, sum+=score2**2; 
		for i in range(len(delta_edge)):
			sum += (deltaE[i][2])**2
			for j in range(len(groundtruth_edge)):
				 if (delta_edge[i] == groundtruth_edge[j]).all() or (delta_edge[i] == groundtruth_edge[j][::-1]).all():
					sum -= (deltaE[i][2])**2
					# print "sum is", sum
					break
		return sum

# le = LearningEngine('/Users/Pragya/Documents/SDL/SLD-C1/le-config.json')
# le.runForOneRate(x, y, 0, 10, 10)