from sklearn.preprocessing import normalize
import random
import numpy as np, subprocess, json, logging, sys
from scipy.io import loadmat, savemat
from sets import Set
import matplotlib.pyplot as plt
from GraphFactory import *
import logging
from datetime import datetime

# logfname = 'LE_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
# self.logger = logging.basicConfig(filename=logfname,filemode='w',level= logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')	
class LearningEngine:
	def __init__(self, configFile, logger):
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
		np.sort(self.deltaEGT[:,0:2],axis=1)
		# self.groundtruth_edge = self.convertToInteger(self.deltaEGT[:,:2])
		self.nodesGT = data["nodes"].flatten()
		self.logger = logger

	def writeGraphResultsToFile(self, results):
		fh = open(self.resultsFolder + "results.txt","w")
		fh.write(results)
		fh.close()

	def runForDifferentRates(self, x, y, spacing, iterations):
		graphVarsForRates = {}
		for rate in self.learningRate:
			graphVars = self.runForOneRate(x, y, spacing, iterations, rate)
			graphVarsForRates[rate] = graphVars
		print graphVarsForRates
		self.writeGraphResultsToFile(graphVarsForRates)
	
	def runWithRestarts(self, x, y, spacing, iterations, learningRate, noOfRestarts):
		graphVarsForRestarts = {}
		for i in range(noOfRestarts):
			graphVars = self.runForOneRate(x, y, spacing, iterations, learningRate)
			graphVarsForRestarts[i] = graphVars
		print graphVarsForRestarts
		self.writeGraphResultsToFile(graphVarsForRestarts)

	def runForOneRate(self, x, y, spacing, iterations, learningRate):
		# sigma = random.uniform(0, 10)
		sigma = 1
		prevSigma = 0
		prevLoss = sys.maxint
		self.logger.warn("Running ML engine for one rate.")
		self.logger.warn("Initial sigma: %d", sigma)
		# [iteration, sigma, loss, learningRate]
		graphVars = [[0, sigma, 0, learningRate]]
		for i in range(iterations):
			self.logger.warn("Iteration: %d", i)
			self.logger.warn("Creating graph 1")
			graphFactory = GraphFactory(x, sigma, spacing, self.nodeFolder1, True)
			graphFactory.createGraph()
			self.logger.warn("Done creating graph 1")
			self.logger.warn("Creating graph 2")
			graphFactory2 = GraphFactory(y, sigma, spacing, self.nodeFolder2, True)
			graphFactory2.createGraph()
			self.logger.warn("Done creating graph 2")
			self.runCAD()
			data = loadmat(self.resultsFolder + 'result.mat')
			deltaE = data["deltaE"]
			nodes = data["nodes"].flatten()
			self.logger.warn("Done loading results. Now comparing nodes")
			print "Nodes ", nodes
			print "GT Nodes", self.nodesGT
			if i > 0 and len(nodes) == len(self.nodesGT) and (nodes == self.nodesGT).all():
				self.logger.warn("Found sigma: %f", sigma)
				self.logger.warn("Exiting")
				print "Found sigma ", sigma
				break
			self.logger.warn("Done comparing. Starting gradientDescent")
			loss, newSigma = self.gradientDescent(sigma, deltaE, learningRate, prevLoss, prevSigma)
			graphVars.append([i+1, newSigma, loss, learningRate])
			if sigma == newSigma:
				self.logger.warn("Sigma didn't change in the last iteration")
				self.logger.warn("Exiting")
				break
			else:
				prevLoss = loss
				prevSigma = sigma
				sigma = newSigma
		print "Graph elements ", graphVars
		# plt.plot(graphVars[:][0],graphVars[:][1], 'o')
		# plt.plot(graphVars[:][0],graphVars[:][2], '+')
		# plt.show()
		return graphVars
	
	def gradientDescent(self, sigma, deltaE, learningRate, prevLoss, prevSigma):
		self.logger.warn("Starting gradientDescent")
		sum = self.lossFunction(deltaE)
		if prevLoss == sys.maxint:
			self.logger.warn("Ending gradientDescent loss %f new sigma %f", sum, sigma + 0.4)
			return sum, sigma + 0.4
		descent = sigma - learningRate * (sum - prevLoss)/(sigma - prevSigma)
		self.logger.warn("Ending gradientDescent loss %f new sigma %f", sum, descent)
		return sum, descent

	#Assuming ranks is an array of nodes. Their position is defined by array index.
	def lossFunctionKamalika(self, ranks, preknownRanks):
		sumOfDistances = 0
		for i, rank in enumerate(ranks):
			j = self.dataSize
			if rank in preknownRanks:
				j = preknownRanks.index(rank)
			sumOfDistances += (i - j)^2
		return 1 - sumOfDistances / (self.dataSize * (self.dataSize ^ 2 - 1))

	def runCAD(self):
		# print "Starting CAD"
		self.logger.warn("Starting CAD")
		cmd = [self.spark, self.CAD, self.CADOptions]
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
		out, err = p.communicate()
		self.logger.warn("Done running CAD")
		# print out

	def convertToInteger(self, deltaEArray):
		deltaEArray[:, 0] = np.asarray(deltaEArray[:, 0]).astype(int)
		deltaEArray[:, 1] = np.asarray(deltaEArray[:, 1]).astype(int)
		return deltaEArray

	def lossFunction(self, deltaE):
		self.logger.warn("Starting loss function")
		edge_dict = {}

		for i in range(len(self.deltaEGT)):
			edge_dict[(self.deltaEGT[i][0],self.deltaEGT[i][1])] = self.deltaEGT[i][2]

		for j in range(len(deltaE)):
			tup = (deltaE[j][0], deltaE[j][1])
			if edge_dict.get(tup, 'None') == 'None':
				edge_dict[tup] = deltaE[j][2]
			else:
				edge_dict[tup] -= deltaE[j][2]

		sqValues = np.asarray(edge_dict.values())**2
		self.logger.warn("Ending loss function")
		return np.sum(sqValues)

	# def lossFunction(self, deltaE):
	# 	self.logger.warn("Starting loss function")
	# 	# print("======================== edge list of deltaE =========================")
	# 	delta_edge = self.convertToInteger(deltaE[:,:2])
	# 	print(delta_edge)
	# 	# print("======================== edge list of groundTruth deltaE =============")
	# 	sum = 0
	# 	# print("=======================================================================")
	# 	# Iterate edge in groundtruthdEdge. If there is same edge in delteEdge, sum+= (score1 - score2)**2; otherwise, sum+=score1**2
	# 	for i in range(len(self.groundtruth_edge)):
	# 		sum += (self.deltaEGT[i][2])**2
	# 		for j in range(len(delta_edge)):
	# 			# print "deltaEdge",i,": ",delta_edge[j]
	# 			# print "deltaGroundTruth", j, ": ",groundtruth_edge[i]
	# 			if (delta_edge[j] == self.groundtruth_edge[i]).all() or (delta_edge[j] == self.groundtruth_edge[i][::-1]).all() :
	# 				# print "true"
	# 				sum += (self.deltaEGT[i][2] - deltaE[j][2])**2
	# 				sum -= (self.deltaEGT[i][2])**2
	# 				# print "sum is", sum
	# 				break		
	# 	# Iterate edge in delteEdge. If there no same edge in groundtruthdEdge, sum+=score2**2; 
	# 	for i in range(len(delta_edge)):
	# 		sum += (deltaE[i][2])**2
	# 		for j in range(len(self.groundtruth_edge)):
	# 			 if (delta_edge[i] == self.groundtruth_edge[j]).all() or (delta_edge[i] == self.groundtruth_edge[j][::-1]).all():
	# 				sum -= (deltaE[i][2])**2
	# 				# print "sum is", sum
	# 				break
	# 	self.logger.warn("Ending loss function with value %f", sum)
	# 	return sum