from sklearn.preprocessing import normalize
import random
import numpy as np, subprocess, json, logging, sys
from scipy.io import loadmat, savemat
from sets import Set
import matplotlib.pyplot as plt
from GraphFactory import *
import logging
from datetime import datetime
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Initializators, Mutators, Crossovers
from pyevolve import Scaling
from pyevolve import Consts
import math
from DataSynthesizer import *
from scipy.optimize import minimize

logfname = 'GA_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
logging.basicConfig(filename=logfname,filemode='w',level= logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')	

def runCAD():
	# print "Starting CAD"
	cmd = [spark, CAD, CADOptions]
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
	out, err = p.communicate()

def lossFunction(deltaE):
		logging.warn("Starting loss function")
		edge_dict = {}

		for i in range(len(deltaEGT)):
			edge_dict[(deltaEGT[i][0],deltaEGT[i][1])] = deltaEGT[i][2]

		for j in range(len(deltaE)):
			tup = (deltaE[j][0], deltaE[j][1])
			if edge_dict.get(tup, 'None') == 'None':
				edge_dict[tup] = deltaE[j][2]
			else:
				edge_dict[tup] -= deltaE[j][2]

		sqValues = np.asarray(edge_dict.values())**2
		logging.warn("Ending loss function")
		return np.sum(sqValues)

def evalFunction(newSigma):
	print "new sigma", newSigma
	print "newSigma[0]", newSigma[0]
	gf = GraphFactory(x, newSigma[0], 1, nodeFolder1, True)
	gf.createGraph()
	gf2 = GraphFactory(y, newSigma[0], 1, nodeFolder2, True)
	gf2.createGraph()
	cmd = [spark, CAD, CADOptions]
	p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
	out, err = p.communicate()
	print out
	data = loadmat(resultsFolder + 'result.mat')
	deltaE = data["deltaE"]
	nodes = data["nodes"].flatten()
	print "Nodes ", nodes
	print "GT Nodes", nodesGT
	if len(nodes) == len(nodesGT) and (nodes == nodesGT).all():
		print "Found sigma ", newSigma
		return 0
	lossValue = lossFunction(deltaE)
	logging.warn("Loss function %f", lossValue)
	logging.warn("Eval function %f", lossValue)
	# return -1 * lossValue
	return lossValue

x, y = sytheticDATA(16)
sig = 1/(2**0.5)
graphFactory = GraphFactory(x, sig, 0, '/home/ldapuser1/code-from-git/SLD-C1/GeneticAlgorithm/', True)
graphFactory.createGraph()
graphFactory2 = GraphFactory(y, sig, 0, '/home/ldapuser1/code-from-git/SLD-C1/GeneticAlgorithm2/', True)
graphFactory2.createGraph()
print "Starting CAD"
cmd = ["/home/ldapuser1/spark-2.0.2-bin-hadoop2.4/bin/spark-submit", "/home/ldapuser1/code-from-git/SLD-C1/CAD.py", "/home/ldapuser1/code-from-git/SLD-C1/options.json"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
output, error = proc.communicate()
print output

configFile = '/home/ldapuser1/code-from-git/SLD-C1/le-config.json'
with open(configFile) as config_file:
	data = json.load(config_file)
	spark = data["spark"]
	CAD = data["CAD"]
	CADOptions = data["CADOptions"]
	nodeFolder1 = data["nodeFolder1"]
	nodeFolder2 = data["nodeFolder2"]
	dataSize = data["totalDataSize"]
	resultsFolder = data["resultsFolder"]
	learningRate = data["learningRate"]
data = loadmat(resultsFolder + 'GTresult.mat')
deltaEGT = data["deltaE"]
np.sort(deltaEGT[:,0:2],axis=1)
nodesGT = data["nodes"].flatten()

# res = minimize(evalFunction, 1.4, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
genome = G1DList.G1DList(1)
genome.setParams(rangemin=0.0, rangemax=1.4)

# Change the initializator to Real values
genome.initializator.set(Initializators.G1DListInitializatorReal)

# Change the mutator to Gaussian Mutator
genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

# Removes the default crossover
genome.crossover.clear()

# The evaluator function (objective function)
genome.evaluator.set(evalFunction)

# Genetic Algorithm Instance
ga = GSimpleGA.GSimpleGA(genome)
ga.setMinimax(Consts.minimaxType["minimize"])

pop = ga.getPopulation()
pop.scaleMethod.set(Scaling.SigmaTruncScaling)

ga.selector.set(Selectors.GRouletteWheel)
ga.setGenerations(4)
ga.setPopulationSize(4)

# Do the evolution
ga.evolve(freq_stats=1)

# Best individual
print ga.bestIndividual()
logging.warn("Done")


