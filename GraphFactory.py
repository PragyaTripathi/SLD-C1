import numpy as np, json, os, scipy.io
from math import exp

class GraphFactory:
	def __init__(self, timeSliceMatrix, sigma, spacing, folderName):
		self.timeSliceMatrix = timeSliceMatrix
		self.sigma = sigma
		self.spacing = spacing
		if not os.path.exists(folderName):
			os.makedirs(folderName)
		self.folderName = folderName

	def createGraph(self):
		self.formGraphFromSubSamples(self.subSampleAndFlatten())

	def subSampleAndFlatten(self):
		matrix = np.array(self.timeSliceMatrix)
		i,j = matrix.shape
		flatArray = []
		mappingToLocation = {}
		index = 0
		for a in range(i/self.spacing):
			for b in range(j/self.spacing):
				average = self.meanOfNonZeroElements(matrix[a*self.spacing:a*self.spacing+self.spacing][:,b*self.spacing:b*self.spacing+self.spacing])
				mappingToLocation[index] = {"latitude": a*self.spacing, "longitude" : b*self.spacing}
				flatArray.append(average)
				index += 1
		json.dump(mappingToLocation, open(self.folderName + "/mappingToLocation.txt",'w'))
		return flatArray

	def formGraphFromSubSamples(self, flatArray):
		arrayLen = len(flatArray)
		p = self.find_divisor(arrayLen)
		for i in range(arrayLen/p):
			for j in range(arrayLen/p):
				print("block matrix [{0}:{1}][{2}:{3}]".format(i*p, i*p+p, j*p, j*p+p))
				blockName = '{0}_{1}'.format(i*p, j*p)
				blockFilename = blockName + '.mat'
				blockMatrix = [[exp(-1 * abs(flatArray[i*p+m] - flatArray[j*p+n])/(2*self.sigma**2)) if i*p+m != j*p+n else 0 \
					for n in range(p)] \
					for m in range(p)]
				scipy.io.savemat(self.folderName + blockFilename, mdict={blockName: blockMatrix})
				with open(self.folderName + "filelist.txt", "a") as blockMatrixIndexfile:
					blockMatrixIndexfile.write(blockFilename + "\n")

	def my_range(self, start, end, step):
		while end <= start:
			yield start
			start += step

	def find_divisor(self, length):
		for i in self.my_range(150, 10, -1):
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