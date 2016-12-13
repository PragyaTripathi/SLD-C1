import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import BlockMatrix, RowMatrix, DistributedMatrix, CoordinateMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.mllib.linalg import Matrices, SparseMatrix, DenseMatrix
from pyspark.sql import SQLContext
from scipy.io import loadmat
from scipy import sparse
import numpy as np, scipy.io as sio, os, re, mmap, math, logging, time, subprocess, gc, itertools as it, copy, functools
from datetime import datetime
from operator import add
from collections import defaultdict

SQUARE_BLOCK_SIZE = 0

def MapperLoadBlocksFromMatFile(filename):
	logging.warn('MapperLoadBlocksFromMatFile started %s ', filename)
	data = loadmat(filename)
	logging.warn('Loaded data')
	print filename
	name = re.search('(\d_\d).mat$', filename, re.IGNORECASE).group(1)
	print name
	# name = max(enumerate(nameMatches))[1]
	G = data[name]
	id = name.split('_')
	n = G.shape[0]
	logging.warn('Before sparse conversion')
	if(not(isinstance(G,sparse.csc_matrix))):
		sub_matrix = Matrices.dense(n, n, G.transpose().flatten())
	else:
		#sub_matrix = Matrices.dense(n,n,np.array(G.todense()).transpose().flatten())
		#SPARSE
		sub_matrix = Matrices.sparse(n,n,G.indptr,G.indices,G.data)
	logging.warn('MapperLoadBlocksFromMatFile Ended')
	return ((id[0], id[1]), sub_matrix)

def EdgesPerBlock(b):
	squareBlockSize = globals()["SQUARE_BLOCK_SIZE"]
	print "squareBlockSize"
	print squareBlockSize
	#logging.warn('EdgesPerBlock started')
	I = b[0][0]
	J = b[0][1]
	print "processing block",I,J,b[1]
	if I>J:
		if isinstance(b[1], SparseMatrix):
		
			logging.warn('Start csc_matrix')
			mat = sparse.csc_matrix((b[1].values, b[1].rowIndices, b[1].colPtrs), shape=(b[1].numRows, b[1].numCols))
			links = sparse.find(mat)
			edges = zip((links[0]+ (I * squareBlockSize)), (links[1]+ (J * squareBlockSize)),links[2])
			logging.warn('End csc_matrix')
		
		else:
			mat = np.array(b[1].toArray())
			if I == J:
				mat = np.triu(mat)
			i,j = np.nonzero(mat)
			values = mat[i,j]
			i = i + I * squareBlockSize
			j = j + J * squareBlockSize
			edges = []		
			for ind in range(len(values)):
				edges.append((i[ind], j[ind], values[ind]))	
			logging.warn('EdgesPerBlock ended')		
	return edges


def RowBlocksDotVec(blocksInRow, vec):
	id = blocksInRow[0]
	blockMatrices = list(blocksInRow[1])
	j = 0
	blockSize = 0
	isSparse = False
	for block in blockMatrices:		
		if j==0:
			blockSize = block.numRows
			# Row vector
			prod = np.zeros(blockSize)
			if isinstance(block, SparseMatrix):
				isSparse = True

		if isSparse:
			b = sparse.csc_matrix((block.values, block.rowIndices, block.colPtrs), shape=(block.numRows, block.numCols))
		else:		
			b = block.toArray()
		subVector = vec[j:(j + blockSize)]
		prod = prod + b.dot(subVector)
		j = j + blockSize
	return (id, prod)

class CAD:
	def __init__(self, squareBlockSize, mainMatrixSize):
		self.inputDir = ""
		self.blocksDir = ""
		self.squareBlockSize = squareBlockSize
		self.mainMatrixSize = mainMatrixSize
		self.tol = 1e-2
		self.epsilon = 0.1
		self.d = 3
		global SQUARE_BLOCK_SIZE
		SQUARE_BLOCK_SIZE = squareBlockSize

		# Spark configs
		self.conf = SparkConf().set("spark.driver.maxResultSize", "0")
		self.conf.set( "spark.akka.frameSize", "2040")
		self.sc = SparkContext(conf=self.conf, appName="Commute time distances ")
		self.sqlContext = SQLContext(self.sc)
		self.minPartitions = 10000

		# Logging vars
		self.runtimeProfiling = False
		loglevel = 'logging.INFO'
		logfname = 'log_'+'size_'+str(mainMatrixSize)+'_BS_'+str(squareBlockSize)+'_minP_'+str(self.minPartitions)+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
		logging.basicConfig(filename=logfname,filemode='w',level= logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')	
		logging.info('Parameters Selected:\n tol:\t%d\n epsilon:\t%d\n dvalu/e:\t%d\n BLOCKS_DIR:\t%s\n PROFILING:\t%s\n',self.tol,self.epsilon,self.d,self.blocksDir,self.runtimeProfiling)

	def difun(self, x, vect):
		squareBlockSize = copy.deepcopy(self.squareBlockSize)
		if( x[0] == x[1]):
			sm = SparseMatrix(squareBlockSize, squareBlockSize, np.linspace(0, squareBlockSize, num = (squareBlockSize+1)), np.linspace(0, squareBlockSize-1, num = squareBlockSize), vect[(x[0]*squareBlockSize):((x[0]+1)*squareBlockSize)])
			return (x,sm)
		else:
			h = sparse.csc_matrix((squareBlockSize,squareBlockSize))
			return (x, Matrices.sparse(squareBlockSize,squareBlockSize,h.indptr,h.indices,h.data))

	def run(self, graphFolder1, graphFolder2):
		adjacency_mat1, edge_list1 = self.loadGraph(graphFolder1)
		adjacency_mat2, edge_list2 = self.loadGraph(graphFolder2)
		
		logging.info('Calling CommuteTimeDistance')
			
		dij1 = self.commuteTimeDistances(edge_list1, adjacency_mat1, self.tol, self.epsilon, self.d)	
		dij2 = self.commuteTimeDistances(edge_list2, adjacency_mat2, self.tol, self.epsilon, self.d)
		
		#Set this threshold value.
		threshold = 10
		deltaE = self.getDeltaE(adjacency_mat1,edge_list1,dij1,adjacency_mat2,edge_list2,dij2)

	def loadGraph(self, graphFolder):
		self.inputDir = graphFolder
		if not os.path.exists(graphFolder + "SparkBlocks/"):
			os.makedirs(graphFolder + "SparkBlocks/")
		self.blocksDir = graphFolder + "SparkBlocks/"
		filelist = self.sc.textFile(graphFolder + 'filelist.txt', minPartitions = self.minPartitions)
		blocks_rdd = filelist.map(MapperLoadBlocksFromMatFile)
		elist_file = graphFolder + 'elist.mat'
		inp = loadmat(elist_file)
		edge_list = inp['elist']
		blocksize = copy.deepcopy(self.squareBlockSize)
		matrixSize = copy.deepcopy(self.mainMatrixSize)
		# number of rows per block, number of columns per block, number of rows in giant matrix, number of columns in giant matrix
		adjacency_mat = BlockMatrix(blocks_rdd, blocksize, blocksize, matrixSize, matrixSize)
		logging.warn('adjacency_mat is created with :\n rows:\t %d\ncols:\t %d \n NumOfRowsPerBlock : \t %d \n NumColsPerBlock:\t %d \n',adjacency_mat.numRows(),adjacency_mat.numCols(),adjacency_mat.rowsPerBlock,adjacency_mat.colsPerBlock)	
		logging.warn('CreateInputsDistributed ended')
		return adjacency_mat, edge_list

	def getdict(self,edge_list,dij):
		combined = np.column_stack(edge_list,dij)
		edgeVSdistance = dict([ ( (x[0],x[1]),x[2] ) for x in combined ])
		edgeVSdistance = defaultdict(lambda : 0.0,edgeVSdistance)
		return edgeVSdistance

	def getDeltaE(self,adjacency_mat1,edge_list1,dij1,adjacency_mat2,edge_list2,dij2):
		'''
		dij : a vector with commute-time distance for each edge in edge_list
		edge_list : a ndarry with shape (NumOfEdges,2)[Usually given as input along with adj mat]
		adjacency_mat : adjacency_matrix of the graph , usually Distributed BLOCK Matrix

		delta e(i,j) = |A2(i,j) - A1(i,j)| * | d2 -d1(i,j)|
		'''
		#difference of adjs, not taken absolute yet.
		deltaA =  adjacency_mat1.subtract(adjacency_mat2)
		#difference in commutedistnaces.
		#assumption is that they both are of the same edge order. We can change that by creating a union of edge lists

		edgeVSdistance1 = self.getdict(edge_list1,dij1)
		edgeVSdistance2 = self.getdict(edge_list2,dij2)

		nonzero_edges, delA_values = self.extractEdgesFromAdjBlock(deltaA)

		nonzero_delA = np.column_stack( (nonzero_edges, np.absolute(delA_values) ))

		# this is np.ndarray with mapping -> edges,delE np.array([4,1,55]) where 4,1 are edge indices, 55 s deltaE
		delE = [ [x[0],x[1], ( x[2] * abs( edgeVSdistance1([(x[0],x[1])]) - edgeVSdistance2([(x(0),x[1])]) ) ) ] for x in nonzero_delA ]

		increasing_edge_index_deltaE = delE[delE[:,2].argsort()]
		# decreasing_edge_index_deltaE = increasing_edge_index_deltaE[::-1]
		deltaE_cumsum = np.cumsum(increasing_edge_index_deltaE[:,2])
		anom_edge_begin = next(x[0] for x in enumerate(deltaE_cumsum) \
			if x[1] > threshold)

		Et = increasing_edge_index_deltaE\
			[anom_edge_begin:len(increasing_edge_index_deltaE), 0:2]
		Et = zip(Et[:, 0], Et[:, 1])
		return Et

	def rudeSolverCMU(self, C, A0, d):
		# C = D0
		logging.warn('RudeSover: BEGIN')
		n = int(C.numRows())
		S0_powers = mult(mult(C,A0),C)
		I = self.diagonalBlockMatrix(np.ones(n),True)
		for i in range(d-1):
			logging.warn('Iter: '+ str(i))
			if i > 0:
				S0_powers =  mult(S0_powers,S0_powers)
				
			C = mult((S0_powers.add(I)),C)
		I.blocks.unpersist()
		logging.warn('RudeSover:End')
		return C

	def commuteTimeDistances(self, elist, A_block_mat, tol, epsilon, d):
		logging.warn('Starting Commute Time Distance')
		logging.warn('\n\n Num blocks = %d', A_block_mat.blocks.count())
		scale = 4
		tolProb = 0.5
		n = A_block_mat.numRows()
		p = SQUARE_BLOCK_SIZE
		
		D0_block_mat, D1_block_mat = self.rowSumD(A_block_mat)
		ChainProduct = self.rudeSolverCMU(D1_block_mat, A_block_mat, d)
		logging.warn('\n\n Starting DEATH TRAP')

		D1_Cprod = mult(D1_block_mat, ChainProduct)
		D1_Cprod.cache()

		all_edges, edge_weights = self.extractEdgesFromAdjBlock(A_block_mat)
		B = ConstructB(all_edges, n)
		m = all_edges.shape[0]
		indices = np.linspace(0, m-1, num = m)
		W = sparse.csc_matrix((edge_weights, (indices, indices)), shape = (m,m))
		w_times_b_T = (W * B).transpose()	
		scale = int(math.ceil(math.log(n,2)) / epsilon)
		# IMPLEMENT THIS 
		del B, W
		del all_edges, edge_weights
		
		eff_res = np.zeros(elist.shape[0])
		# --------------------------------
		#d0_minus_a0 = subtractBlockMatrices(D0_block_mat, A_block_mat)
		d0_minus_a0 = D0_block_mat.subtract(A_block_mat)
		logging.warn('\n\n Starting DEATH TRAP 2')
		ChainProduct2 = mult(D1_Cprod,d0_minus_a0)

		logging.warn('Before unpersist')
		d0_minus_a0.blocks.unpersist()
		D0_block_mat.blocks.unpersist()
		D1_block_mat.blocks.unpersist()
		A_block_mat.blocks.unpersist()
		del d0_minus_a0
		del D0_block_mat, D1_block_mat, A_block_mat
		logging.warn('After unpersist: DELETED objects')

		for j in range(scale):
			logging.warn('Scale iteration: ' + str(j))		
			
			ons = np.random.choice([-1, 1], size=(m,), p=[tolProb, 1-tolProb])
			ons = ons / ((scale ** 0.5) *1.0)

			y = w_times_b_T.dot(ons)
			z = self.exactSolverCMU(D1_Cprod, y, ChainProduct2, tol)
			eff_res = eff_res + (z[elist[:,0]] - z[elist[:,1]]) ** 2

		logging.warn('Out of scale and getting out of commute time distance !!')
		return eff_res

	def extractEdgesFromAdjBlock(self, block_mat):
		#logging.warn('ExtractEdgesFromAdjBlock started')
		
		all_edges_weights_rdd = block_mat.blocks.flatMap(EdgesPerBlock)
		temp = all_edges_weights_rdd.collect()
		temp = np.array(temp)
		#logging.warn('Extract edgesADJ temp done')
		
		all_edges_weights = temp[np.lexsort((temp[:, 0], temp[:, 1]))]
		all_edges = all_edges_weights[:, 0:2]
		all_weights = all_edges_weights[:, 2]
		#logging.warn('ExtractEdgesFromAdjBlock Done')
		return all_edges, all_weights

	def exactSolverCMU(self, D1_Cprod, b0, ChainProduct2, tolerance):
		q = -int(math.ceil(math.log(tolerance)))
		Chi = self.matrixVectorMultiply(D1_Cprod,b0)
		n = int(ChainProduct2.numRows())
		y = Vectors.zeros(n)
		for k in range(q-1):
			temp = self.matrixVectorMultiply(ChainProduct2,y) 
			y = y - temp + Chi
		logging.warn('ExactSover: Done returning')
		return y

	def rowSumD(self, block_mat):
		logging.warn('RowSumD started')
		n = block_mat.numRows()
		D_array = self.matrixVectorMultiply(block_mat,np.ones(n))
		logging.warn('D_array size = %d', len(D_array))
		print "\n\n\n D: ", D_array[0:100] 
		D = self.diagonalBlockMatrix(D_array,True)
		D_array = D_array ** (-0.5)
		D_array[D_array == np.inf] = 0 
		logging.warn('D1_array size = %d', len(D_array))

		D1 = self.diagonalBlockMatrix(D_array,True)
		logging.warn('RowSumD ended')
		return D, D1

	def diagonalBlockMatrix(self, diag, dense = False):
	    n = len(diag)
	    p = self.squareBlockSize
	    num_blocks = n/p
	    blockids = self.sc.parallelize(it.product(xrange(num_blocks),repeat=2))
	    block_rdd = blockids.map(lambda x: self.difun(x,diag))
	    return BlockMatrix(block_rdd,p,p,n,n)

	def matrixVectorMultiply(self, mat, vec):
		completeRows = mat.blocks.map(lambda x: (x[0][0], x[1])).groupByKey()
		productBlocks = completeRows.map(lambda x: RowBlocksDotVec(x, vec))
		product_list = productBlocks.collect()

		n = mat.numRows()
		p = self.squareBlockSize
		product = np.zeros(n)
		
		for i in range(len(product_list)):
			id = product_list[i][0]
			product[(id*p) : ((id+1)*p)] = product_list[i][1]
			
		return product

if __name__=='__main__':
	cad = CAD(128, 256)
	cad.run('/home/ldapuser1/code-from-git/SLD-C1/Test2/', '/home/ldapuser1/code-from-git/SLD-C1/TestWithAnomalies2/')
