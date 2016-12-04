import numpy as np, json, os, scipy.io, shutil
from math import exp

class GraphFactory:
	def __init__(self, timeSliceMatrix, sigma, spacing, folderName, isSynthetic = False):
		self.timeSliceMatrix = timeSliceMatrix
		self.sigma = sigma
		self.spacing = spacing
		shutil.rmtree(folderName, ignore_errors=True)
		os.makedirs(folderName)
		self.folderName = folderName
		self.isSynthetic = isSynthetic

	def createGraph(self):
		if self.isSynthetic:
			self.formGraphFromFlattenedArray(self.timeSliceMatrix)
		else:
			self.formGraphFromFlattenedArray(self.subSampleAndFlatten())

	# Handle Multivariate later
	def subSampleAndFlatten(self):
		if self.spacing == 0:
			self.spacing = 1
		matrix = np.array(self.timeSliceMatrix)
		i,j = matrix.shape
		flatArray = []
		mappingToLocation = []
		index = 0
		for a in range(i/self.spacing):
			for b in range(j/self.spacing):
				average = self.meanOfNonZeroElements(matrix[a*self.spacing:a*self.spacing+self.spacing][:,b*self.spacing:b*self.spacing+self.spacing])
				mappingToLocation.append([a*self.spacing, b*self.spacing])
				flatArray.append(average)
				index += 1
		scipy.io.savemat(self.folderName + "/mappingToLocation.mat", mdict={"Location": mappingToLocation})
		return flatArray

	def formGraphFromFlattenedArray(self, flatArray):
		arrayLen = np.asarray(flatArray).shape[0]
		print arrayLen
		p = self.find_divisor(arrayLen)
		print p
		edgeList = []
		for i in range(arrayLen/p):
			for j in range(arrayLen/p):
				print("block matrix [{0}:{1}][{2}:{3}]".format(i*p, i*p+p, j*p, j*p+p))
				blockName = '{0}_{1}'.format(i, j)
				blockFilename = blockName + '.mat'
				blockMatrix = []
				for m in range(p):
					subArray = []
					for n in range(p):
						value = exp(-1 * self.sumOfEuclidean(flatArray, i*p+m, j*p+n)/(2*self.sigma**2)) if i*p+m != j*p+n else 0
						if value != 0:
							edgeList.append([i*p+m, j*p+n])
						subArray.append(value)
					blockMatrix.append(subArray)    
				# print(blockMatrix)
				scipy.io.savemat(self.folderName + blockFilename, mdict={blockName: blockMatrix})
				with open(self.folderName + "filelist.txt", "a") as blockMatrixIndexfile:
					blockMatrixIndexfile.write(self.folderName + blockFilename + "\n")
		scipy.io.savemat(self.folderName + "/elist.mat", mdict={"elist": edgeList})

	def sumOfEuclidean(self, flatArray, index1, index2):
		sum = 0
		for i in range(np.asarray(flatArray[0]).size):
			sum += (flatArray[index1][i] - flatArray[index2][i])**2
		return sum

	def my_range(self, start, end, step):
		while end <= start:
			yield start
			start += step

	def find_divisor(self, length):
		for i in self.my_range(length-1, 10, -1):
			if length % i == 0 and i % 4 == 0:
				return i
		return length

	# Helper function for finding mean of non zero elements
	def meanOfNonZeroElements(self, ndarray):
		sum = 0
		count = 0
		for row in ndarray:
			for e in row:
				if e > 0:
					sum += e
					count += 1
		return sum/count if count > 0 else 0
