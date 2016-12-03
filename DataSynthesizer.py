import numpy as np
from random import randint
from GraphFactory import *

def sythensizeData(mu, sigma, size_x, size_y):
	s = np.random.normal(mu, sigma, size_x*size_y)
	return np.array([[s[size_y*i+j]for j in range(size_y)] for i in range(size_x)])

def changeNode(num, matrix, times):
	x_position = np.random.randint(0, matrix.shape[0], size=num)
	y_position = np.random.randint(0, matrix.shape[1], size=num)
	for i in range(num):
		matrix[x_position[i]][y_position[i]] = matrix[x_position[i]][y_position[i]]*times
	return matrix, [x_position, y_position]

matrix = sythensizeData(0,1,120,120)
print(matrix)
newmatrix, anomalousNodes = changeNode(10, matrix, 3)
print(newmatrix)
print(anomalousNodes)

graphFactory = GraphFactory(matrix, 1, 4, '/Users/Pragya/Documents/SDL/SLD-C1/SynesizedData/')
graphFactory.createGraph()
graphFactory2 = GraphFactory(newmatrix, 1, 4, '/Users/Pragya/Documents/SDL/SLD-C1/SynesizedDataWithAnomalies/')
graphFactory2.createGraph()