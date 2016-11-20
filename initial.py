import scipy.io, numpy as np
from GraphFactory import *

x = scipy.io.loadmat('/Users/Pragya/Documents/SDL/SLD-C1/temColFormat.mat')
jan1994Index = convertDateToIndex(1996,1) ## January, 1994
matrix = np.array(x['tem'])
ndarray = [[0 if e[jan1994Index] == -9999 else e[jan1994Index] for e in row] \
	for row in matrix]
graphFactory = GraphFactory(ndarray, 1, 4, '/Users/Pragya/Documents/SDL/SLD-C1/January1996/')
graphFactory.createGraph()
# latitude, longitude, month = matrix.shape
# p = find_divisor(latitude, longitude)
# index = 0
# for i in range(latitude/p):
# 	for j in range(longitude/p):
# 		print("block matrix [{0}:{1}][{2}:{3}]".format(i*p, i*p+p, j*p, j*p+p))
# 		smallMatrix = matrix[i*p:i*p+p][:,j*p:j*p+p]
# 		print smallMatrix.shape
# 		# smallMatrix = spatialSubSampling(jan1994Index, smallMatrix, 4)
# 		flatArray = subSampling(ndarray, 4)
# 		# smallMatrix = formGraphFromSubSamples(smallMatrix, 1) ## Taking sigma as 1
# 		# smallMatrix = maintainSparsity(smallMatrix)
# 		# blockName = '{0}_{0}'.format(p*p*index/16)
# 		# blockFilename = blockName + '.mat'
# 		# scipy.io.savemat(blockFilename, mdict={blockName: adjacencyMatrix(smallMatrix)})
# 		# with open("filelist.txt", "a") as blockMatrixIndexfile:
# 		# 	blockMatrixIndexfile.write(blockFilename + "\n")
# 		# index += 1
