import scipy.io
import numpy as np
from math import exp
from fractions import gcd

# Converts a large matrix of size n x m to n/spacing x m/spacing.
# spacing helps to divide large matrix into smaller chunks of spacing x spacing 
# size and we take mean of this smaller chunk and replace spacing x spacing matrix. 
def spatialSubSampling(timeIndex, ndarray, spacing):
	ndarray = [[0 if e[timeIndex] == -9999 else e[timeIndex] for e in row] \
		for row in ndarray]
	matrix = np.array(ndarray)
	print matrix.shape
	i,j = matrix.shape
	return [[meanOfNonZeroElements(matrix[a*spacing:a*spacing+spacing][:,b*spacing:b*spacing+spacing]) \
		for b in range(j/spacing)] \
		for a in range(i/spacing)]

# A little function to map year and month to index in matrix.
def convertDateToIndex(year, month):
	if year < 1982 or year > 2002:
		raise ValueError('Year not supported. Data available only for years between(inclusive) 1982 and 2002')
	if month < 0 or month > 12 or type(month) != int:
		raise ValueError('Unrecognized month number. Please provide valid month number (1-12)')
	return (year - 1982) * 12 + month - 1 # Subtract by 1 to account for python range

# Returns upper triangular matrix. The diagonal elements and lower triangular 
# matrix are zeroes.
def formGraphFromSubSamples(subSamples, sigma):
	onedarray= reduce(lambda list, mat: list + [row for row in mat], subSamples)
	return [[exp(-(onedarray[i] - onedarray[j])/(2*sigma**2)) if i < j else 0 \
		for j in range(len(onedarray))] \
		for i in range(len(onedarray))]

def maintainSparsity(ndarray):
	return [[0 if rowElem != max(row) else rowElem for rowElem in row] \
		for row in ndarray]

# Returns adjacency matrix with all diagonal elements set to zero and upper triangular
# matrix mirroring lower triangular matrix.
# Works only if ndarray's diagonal and lower triangular matrix has all zeros.
def adjacencyMatrix(ndarray):
	return np.array(ndarray) + np.transpose(ndarray)

# Helper function for finding mean of non zero elements
def meanOfNonZeroElements(ndarray):
	sum = 0
	count = 0
	for row in ndarray:
		for e in row:
			if e > 0:
				sum += e
				count += 1
	return sum/count if count > 0 else 0

def my_range(start, end, step):
	while end <= start:
	    yield start
	    start += step

def find_divisor(row, column):
	for i in my_range(150, 10, -1):
		if row % i == 0 and column % i == 0 and i % 4 == 0:
			return i
	return 1

x = scipy.io.loadmat('/Users/Pragya/Documents/SDL/SLD-C1/temColFormat.mat')
jan1994Index = convertDateToIndex(1995,1) ## January, 1994
matrix = np.array(x['tem'])
latitude, longitude, month = matrix.shape
p = find_divisor(latitude, longitude)
index = 0
for i in range(latitude/p):
	for j in range(longitude/p):
		print("block matrix [{0}:{1}][{2}:{3}]".format(i*p, i*p+p, j*p, j*p+p))
		smallMatrix = matrix[i*p:i*p+p][:,j*p:j*p+p]
		print smallMatrix.shape
		smallMatrix = spatialSubSampling(jan1994Index, smallMatrix, 4)
		smallMatrix = formGraphFromSubSamples(smallMatrix, 1) ## Taking sigma as 1
		smallMatrix = maintainSparsity(smallMatrix)
		blockName = '{0}_{0}'.format(p*p*index/16)
		blockFilename = blockName + '.mat'
		scipy.io.savemat(blockFilename, mdict={blockName: adjacencyMatrix(smallMatrix)})
		with open("filelist.txt", "a") as blockMatrixIndexfile:
			blockMatrixIndexfile.write(blockFilename + "\n")
		index += 1
