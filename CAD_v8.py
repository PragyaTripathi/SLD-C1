import sys
print sys.version
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import BlockMatrix, RowMatrix, DistributedMatrix, CoordinateMatrix, IndexedRowMatrix, IndexedRow
from pyspark.mllib.linalg import DenseVector, Vectors
from pyspark.mllib.linalg import Matrices, SparseMatrix, DenseMatrix
# https://spark.apache.org/docs/1.4.0/api/java/org/apache/spark/mllib/linalg/SparseMatrix.html
from pyspark.sql import SQLContext
import numpy as np
from scipy.io import loadmat
from scipy import sparse
import scipy.io as sio
from datetime import datetime
import os, re
import mmap
import math
import logging
import time
import subprocess
import gc
from operator import add
import itertools as it
from collections import defaultdict

#Multiply functions--------------------------------------------------------------------------------------
def prod(s1,s2):
	a = sparse.csc_matrix((s1[1].values,s1[1].rowIndices,s1[1].colPtrs),shape=(s1[1].numRows,s1[1].numCols))
	b = sparse.csc_matrix((s2[1].values,s2[1].rowIndices,s2[1].colPtrs),shape=(s2[1].numRows,s2[1].numCols))
	return a.dot(b)

def affectRight(a,numBlocks):
	outBlocks = []
	for i in range(numBlocks):
		outBlocks.append(((i,a[0][0],a[0][1]),a))
	return outBlocks      

def affectLeft(a,numBlocks):
	outBlocks = []
	for i in range(numBlocks):
		outBlocks.append(((a[0][0],a[0][1],i),a))
	return outBlocks

def mult(A,B):
	#-------LOG
	logging.warn("Multiplication started")		
	blockcount = A.blocks.getNumPartitions()
	logging.warn("A part count")
	logging.warn(blockcount)
	blockcount = B.blocks.getNumPartitions()
	logging.warn("B part count")
	logging.warn(blockcount)
	#-----LOG

	# If dense, just call the inbuilt function.
	if(isinstance(A.blocks.first()[1],DenseMatrix) or isinstance(B.blocks.first()[1],DenseMatrix)):
		return A.multiply(B)
	#sparse ? Then continue the madness
	
	N = A.numRows()
	p = SQUARE_BLOCK_SIZE
	num_blocks = N/p
	
	aleft = A.blocks.flatMap(lambda x: affectLeft(x,num_blocks))
	bright = B.blocks.flatMap(lambda x: affectRight(x,num_blocks))
	both = aleft.union(bright)
	indi = both.reduceByKey(lambda a,b: prod(a,b))
	map = indi.map(lambda x: ((x[0][0],x[0][2]), x[1]) )
	pr = map.reduceByKey(add)
	brd  = pr.map(lambda x : ( (x[0][0],x[0][1]), Matrices.sparse(p,p,x[1].indptr,x[1].indices,x[1].data ) ) )
	C = BlockMatrix(brd,p,p,N,N)
	return C 

#--Multiplication end---------------------------------------------------------------


#SPARSE COMPATIBLE: IN PROGRESS
def CommuteTimeDistances(elist, A_block_mat, tol, epsilon, d):
	logging.warn('Starting Commute Time Distance')
	logging.warn('\n\n Num blocks = %d', A_block_mat.blocks.count())
	scale = 4
	tolProb = 0.5
	n = A_block_mat.numRows()
	p = SQUARE_BLOCK_SIZE
	
	D0_block_mat, D1_block_mat = RowSumD(A_block_mat)
	ChainProduct = RudeSolverCMU(D1_block_mat, A_block_mat, d)
	logging.warn('\n\n Starting DEATH TRAP')

	D1_Cprod = mult(D1_block_mat, ChainProduct)
	D1_Cprod.cache()

	all_edges, edge_weights = ExtractEdgesFromAdjBlock(A_block_mat)
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
		z = ExactSolverCMU(D1_Cprod, y, ChainProduct2, tol)
		eff_res = eff_res + (z[elist[:,0]] - z[elist[:,1]]) ** 2

	logging.warn('Out of scale and getting out of commute time distance !!')
	return eff_res

#SPARSE COMPATIBLE
def ExactSolverCMU(D1_Cprod, b0, ChainProduct2, tolerance):
	q = -int(math.ceil(math.log(tolerance)))
	Chi = MatrixVectorMultiply(D1_Cprod,b0)
	n = int(ChainProduct2.numRows())
	y = Vectors.zeros(n)
	for k in range(q-1):
		temp = MatrixVectorMultiply(ChainProduct2,y) 
		y = y - temp + Chi
	logging.warn('ExactSover: Done returning')
	return y

#SPARSE COMPATIBLE
def RudeSolverCMU(C, A0, d):
	# C = D0
	logging.warn('RudeSover: BEGIN')
	n = int(C.numRows())
	S0_powers = mult(mult(C,A0),C)
	I = DiagonalBlockMatrix(np.ones(n),True)
	for i in range(d-1):
		logging.warn('Iter: '+ str(i))
		if i > 0:
			S0_powers =  mult(S0_powers,S0_powers)
			
		C = mult((S0_powers.add(I)),C)
	I.blocks.unpersist()
	logging.warn('RudeSover:End')
	return C
	
def MatrixVectorMultiply(mat, vec):
	completeRows = mat.blocks.map(lambda x: (x[0][0], x[1])).groupByKey()
	productBlocks = completeRows.map(lambda x: RowBlocksDotVec(x, vec))
	product_list = productBlocks.collect()

	n = mat.numRows()
	p = SQUARE_BLOCK_SIZE
	product = np.zeros(n)
	
	for i in range(len(product_list)):
		id = product_list[i][0]
		product[(id*p) : ((id+1)*p)] = product_list[i][1]
		
	return product

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

'''
New fun ended
'''
# ASSUMPTION: both are square matrices: n x n
#SPARSE COMPATIBLE :

def subtractBlockMatrices(left, right):
	logging.warn('subtractBlockMatrices started')
	n = right.numRows()
	'''
	negI_sm = SparseMatrix(n, n, np.linspace(0, n, num = (n+1)), np.linspace(0, n-1, num = n), -1*np.ones(n))
	negI = BlockMatrix(sc.parallelize([((0, 0), negI_sm)]), SQUARE_BLOCK_SIZE, SQUARE_BLOCK_SIZE)
	'''
	negI = DiagonalBlockMatrix(-1*np.ones(n),True)
	#negative_right = right.multiply(negI)
	negative_right = mult(right,negI)
	result = negative_right.add(left)
	
	# negI.blocks.unpersist()
	logging.warn('subtractBlockMatrices ended')
	return result

def TimeAndPrint(t1, message):
	tD = (datetime.now() - t1).total_seconds()
	t1 = datetime.now()
	print message," | elapsed time = ", tD, "\n\n"
	return t1

def BlockMatInfoString(mat):
	return str(mat.numRows()) + ", " + str(mat.numCols()) + ":: " + \
	 str(mat.rowsPerBlock) + ", " + str(mat.colsPerBlock) + ":: " + \
	 str(mat.numRowBlocks) + ", " + str(mat.numColBlocks)

# WARNING: collect on edge_weights can run out of memory in case of a dense graph
# Return square root of edge weights accoring to Koutis code
#SPARSE COMPATIBLE: IN PROGRESS
def ExtractEdgesFromAdjBlock(block_mat):
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

#SPARSE COMPATIBLE :TESTING NOT DONE
def EdgesPerBlock(b):
	#logging.warn('EdgesPerBlock started')
	I = b[0][0]
	J = b[0][1]
	print "processing block",I,J,b[1]
	if I>J:
		if isinstance(b[1], SparseMatrix):
		
			logging.warn('Start csc_matrix')
			mat = sparse.csc_matrix((b[1].values, b[1].rowIndices, b[1].colPtrs), shape=(b[1].numRows, b[1].numCols))
			links = sparse.find(mat)
			edges = zip((links[0]+ (I * SQUARE_BLOCK_SIZE)), (links[1]+ (J * SQUARE_BLOCK_SIZE)),links[2])
			logging.warn('End csc_matrix')
		
		else:
			mat = np.array(b[1].toArray())
			if I == J:
				mat = np.triu(mat)
			i,j = np.nonzero(mat)
			values = mat[i,j]
			i = i + I * SQUARE_BLOCK_SIZE
			j = j + J * SQUARE_BLOCK_SIZE
			edges = []		
			for ind in range(len(values)):
				edges.append((i[ind], j[ind], values[ind]))	
			logging.warn('EdgesPerBlock ended')		
	return edges

#SPARSE COMPATIBLE
def ConstructB(all_edges, n):
	logging.warn('ConstructB started')
	m = all_edges.shape[0]
	data = np.array([np.ones(m), -1*np.ones(m)]).flatten()
	row = np.array([np.linspace(0, m-1, num = m), np.linspace(0, m-1, num = m)]).flatten()
	col = all_edges.transpose().flatten()
	mat = sparse.csc_matrix((data, (row, col)), shape=(m, n))
	logging.warn('ConstructB ended')
	return mat

# ASSUMPTION: Square block size must be <= n
# D: row_sums 
# D1: row_sums ^ (-0.5)
#SPARSE COMPATIBLE: 
def RowSumD(block_mat):
	logging.warn('RowSumD started')
	n = block_mat.numRows()
	D_array = MatrixVectorMultiply(block_mat,np.ones(n))
	logging.warn('D_array size = %d', len(D_array))
	print "\n\n\n D: ", D_array[0:100] 
	D = DiagonalBlockMatrix(D_array,True)
	D_array = D_array ** (-0.5)
	D_array[D_array == np.inf] = 0 
	logging.warn('D1_array size = %d', len(D_array))

	D1 = DiagonalBlockMatrix(D_array,True)
	logging.warn('RowSumD ended')
	return D, D1

def DiagonalBlockMatrix(diag, dense = False):
    n = len(diag)
    p = SQUARE_BLOCK_SIZE
    num_blocks = n/p
    blockids = sc.parallelize(it.product(xrange(num_blocks),repeat=2))
    block_rdd = blockids.map(lambda x: difun(x,diag))
    return BlockMatrix(block_rdd,p,p,n,n)

def difun(x,vect):
    if( x[0] == x[1]):
        sm = SparseMatrix(SQUARE_BLOCK_SIZE, SQUARE_BLOCK_SIZE, np.linspace(0, SQUARE_BLOCK_SIZE, num = (SQUARE_BLOCK_SIZE+1)), np.linspace(0, SQUARE_BLOCK_SIZE-1, num = SQUARE_BLOCK_SIZE), vect[(x[0]*SQUARE_BLOCK_SIZE):((x[0]+1)*SQUARE_BLOCK_SIZE)])
        return (x,sm)
    else:
        h = sparse.csc_matrix((SQUARE_BLOCK_SIZE,SQUARE_BLOCK_SIZE))
        return (x, Matrices.sparse(SQUARE_BLOCK_SIZE,SQUARE_BLOCK_SIZE,h.indptr,h.indices,h.data))

def CreateInputs(input_case):
	data_file = '/u/vparames/TESTS/3/test-commute-dist-' + str(input_case) + '.mat'
	inp = loadmat(data_file)
	adj_mat = inp['G']
	edge_list = inp['elist']
	n = adj_mat.shape[0]
	sm = Matrices.dense(n, n, adj_mat.transpose().flatten())
	adjacency_mat = BlockMatrix(sc.parallelize([((0, 0), sm)]), SQUARE_BLOCK_SIZE, SQUARE_BLOCK_SIZE)

	return adjacency_mat, edge_list 

#SPARSE COMPATIBLE:
def MapperLoadBlocksFromMatFile(filename):
	logging.warn('MapperLoadBlocksFromMatFile started %s ', filename)
	data = loadmat(INPUT_DIR + filename)
	logging.warn('Loaded data')
	name = filename[:-4]
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

#SPARSE compatible
def SplitMatrixInBlocks(adj_mat, block_size):
	logging.warn('SplitMatrixInBlocks started')
	n = adj_mat.shape[0]
	# This in only in 1 dimension. Actual number of blocks = num_blocks^2
	num_blocks = n / block_size
	if not isinstance(num_blocks, (int, long)):
		print "\n\n\n ======\n ERROR: Block size should divide size of matrix \n"
		quit()
	filelist = open(BLOCKS_DIR + 'filelist.txt', "w")
	for i in range(num_blocks):
		start_row = i * block_size
		end_row = (i+1) * block_size
		if (end_row > n):
			print "Uneven blocks alert !!!"
			end_row = None
		for j in range(num_blocks):
			start_col = j * block_size
			end_col = (j+1) * block_size
			if(end_col > n):
				print "Uneven blocks alert !!!"
				end_col = None
			G = adj_mat[start_row:end_row, start_col:end_col]
			#if(isinstance(G,sparse.csc_matrix) and len(G.nonzero()[0])==0):
			#	continue
			block_id = np.array([i,j])
			if (end_row == None):
				print "Uneven blocks alert !!!"
				end_row = n
			if(end_col == None):
				print "Uneven blocks alert !!!"
				end_col = n
			block_endpoints = [start_row, end_row, start_col, end_col]
			file_name = BLOCKS_DIR + str(i) + '_' + str(j) + '.mat'
			filelist.write("%s\n" % file_name)
			sio.savemat(file_name, {"G":G, "block_id":block_id, "block_endpoints":block_endpoints})
	filelist.close()
	logging.warn('SplitMatrixInBlocks Ended')

#SPARSE compatible
def CreateInputsDistributed(input_case, block_size = 3):
	logging.warn('CreateInputsDistributed started')
	#data_file = '/u/vparames/TESTS/3/test-commute-dist-' + str(input_case) + '.mat'
	if ("GJan" in str(input_case)):
		data_dir = '/u/vparames/TESTS/climate_precip/'
		logging.warn('elist laoding :started')	
		e_file = data_dir + str(input_case) + '-elist.mat'
		inp2 = loadmat(e_file)
		edge_list = inp2['elist']
		logging.warn('elist laoding :done')
		data_file = data_dir + str(input_case)+'.mat'
		inp = loadmat(data_file)
		adj_mat = inp['G']
	else:
		data_file = INPUT_DIR+str(input_case)+'.mat'
		inp = loadmat(data_file)
		adj_mat = inp['G']
		edge_list = inp['elist']
	logging.warn('Mat file loading Done ! ')
	dense = sparse.issparse(adj_mat)
	print "adj shape",adj_mat.shape
	'''
	-----------------------------------------------
	lines added for v4
	get the closest square size that will fit with the given block size
	----------------------------------------------------
	'''
	num_blocks = int(math.ceil(adj_mat.shape[0]/(block_size*1.0)))	
	squared_shape = (block_size * num_blocks)
	original_shape = adj_mat.shape[0] 
	if(not(isinstance(adj_mat,sparse.csc_matrix))):
		adj_mat = np.pad(adj_mat,(0,squared_shape-original_shape),'constant',constant_values=(0))
	else:
		#BUG FIX:Earlier version
		# adj_mat = sparse.csc_matrix((adj_mat.data,adj_mat.nonzero()), shape =(squared_shape,squared_shape))
		ptr = sparse.find(adj_mat) #ptr[0] - indices, ptr[1] -indptr , ptr[2]-data
		adj_mat = sparse.csc_matrix((ptr[2],(ptr[0],ptr[1])), shape =(squared_shape,squared_shape))
	print "adj matrix reshaped to ",adj_mat.shape
	SplitMatrixInBlocks(adj_mat, block_size)
	filelist = sc.textFile(BLOCKS_DIR + 'filelist.txt', minPartitions = minP)
	# filelist = sc.textFile(BLOCKS_DIR + 'filelist.txt')
	blocks_rdd = filelist.map(MapperLoadBlocksFromMatFile)
	# adjacency_mat = BlockMatrix(blocks_rdd, block_size, block_size)
	adjacency_mat = BlockMatrix(blocks_rdd, block_size, block_size, adj_mat.shape[0], adj_mat.shape[1])
	logging.warn('adjacency_mat is created with :\n rows:\t %d\ncols:\t %d \n NumOfRowsPerBlock : \t %d \n NumColsPerBlock:\t %d \n',adjacency_mat.numRows(),adjacency_mat.numCols(),adjacency_mat.rowsPerBlock,adjacency_mat.colsPerBlock)	
	logging.warn('CreateInputsDistributed ended')
	return adjacency_mat, edge_list 

def logBlockMat(adjMat, name= 'UnnamedBlockMatrix'):
	
	if (mode =='INFO' or mode == 'info'):
		logging.warn('Details for BlockMatrix: \n Name:\t %s \n Rows:\t %d \n Cols:\t %d \n NumOfRowsPerBlock : \t %d \n NumColsPerBlock:\t %d \n ',name,adjMat.numRows(),adjMat.numCols(),adjMat.rowsPerBlock,adjMat.colsPerBlock)
		logging.warn('Values:\t %s \n',adjMat.toLocalMatrix())
	else:
		return
	
def logVariable(varible,name='unnamed_variable'):
	if (mode =='INFO' or mode == 'info'):
		logging.warn('name:\t %s \n values: \t %s \n',name,varible)

def getdict(edge_list,dij):
	combined = np.column_stack(edge_list,dij)
	edgeVSdistance = dict([ ( (x[0],x[1]),x[2] ) for x in combined ])
	edgeVSdistance = defaultdict(lambda : 0.0,edgeVSdistance)
	return edgeVSdistance

def GetdeltaE(adjacency_mat1,edge_list1,dij1,adjacency_mat2,edge_list2,dij2,threshold):
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

	edgeVSdistance1 = getdict(edge_list1,dij1)
	edgeVSdistance2 = getdict(edge_list2,dij2)

	nonzero_edges, delA_values = ExtractEdgesFromAdjBlock(deltaA)

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

def LoadGraph(rowsPerBlock, colsPerBlock, totalRows, totalColumns):
	filelist = sc.textFile(INPUT_DIR + 'filelist.txt', minPartitions = 18)
	blocks_rdd = filelist.map(MapperLoadBlocksFromMatFile)
	elist_file = INPUT_DIR + 'elist.mat'
	inp = loadmat(elist_file)
	edge_list = inp['elist']
	# number of rows per block, number of columns per block, number of rows in giant matrix, number of columns in giant matrix
	adjacency_mat = BlockMatrix(blocks_rdd, rowsPerBlock, colsPerBlock, totalRows, totalColumns)
	logging.warn('adjacency_mat is created with :\n rows:\t %d\ncols:\t %d \n NumOfRowsPerBlock : \t %d \n NumColsPerBlock:\t %d \n',adjacency_mat.numRows(),adjacency_mat.numCols(),adjacency_mat.rowsPerBlock,adjacency_mat.colsPerBlock)	
	logging.warn('CreateInputsDistributed ended')
	print adjacency_mat, edge_list
	return adjacency_mat, edge_list


if __name__=='__main__':
	
	#----------------Create new Spark config---------------------------------------------
	
	conf = SparkConf().set("spark.driver.maxResultSize", "0")
	conf.set( "spark.akka.frameSize", "2040")
	sc = SparkContext(conf=conf, appName="Commute time distances ")
	sqlContext = SQLContext(sc)
	
	# -----------PARAMETERS------------------------------------------------------------------
	INPUT_DIR = '/home/ldapuser1/code-from-git/SLD-C1/Test2/'
	BLOCKS_DIR = '/home/ldapuser1/code-from-git/SLD-C1/Test2/SparkBlocks/'

	#----------------------------------------------------------------
	
	PREFIX_CHAIN_PROD = 'PROD/chain-prod-'
	USE_SAVED_CHAIN_PRODUCT = False

	RUNTIME_PROFILING_ONLY = False
	tol = 1e-2  # 1e-4
	epsilon = 0.1  # 1e-1
	d = 3
	np.random.seed(0) #comment this if u wish to truly randomize

	minP = 10000 ## Minimum number of partitions
	SQUARE_BLOCK_SIZE = 10000 ## Size of each input block
	
	#NOTE:
	# We can use Subtract method from spark instead
	# of our own multiplication based implementation.
		
	#---------------------------LOGGING----------------------------------------------
	if len(sys.argv)<2:
		#mode='WARNING'
		mode = 'INFO'
	else:
		mode=sys.argv[1].upper()
	
	loglevel = 'logging.'+mode
	# Q: What is test_cases?
	# logfname = 'log_'+'size_'+str(test_cases[0])+'_BS_'+str(SQUARE_BLOCK_SIZE)+'_minP_'+str()+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
	logfname = 'log_'+'size_'+str(0)+'_BS_'+str(SQUARE_BLOCK_SIZE)+'_minP_'+str()+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
	logging.basicConfig(filename=logfname,filemode='w',level= logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')	
	logging.info('Parameters Selected:\n tol:\t%d\n epsilon:\t%d\n dvalu/e:\t%d\n BLOCKS_DIR:\t%s\n PROFILING:\t%s\n',tol,epsilon,d,BLOCKS_DIR,RUNTIME_PROFILING_ONLY)	
	#----------------------------------------------------------------------------------
	
	logging.info('Calling:: CreateInputs')
	

	
	'''
	LoadGraph : You can write this , I can also help later.

	LoadGraph function should return adjacency_mat1 and edge_list in this way.

	>>>adjacency_mat1
	<pyspark.mllib.linalg.distributed.BlockMatrix object at 0x7ff2400e2790>

	>>> type(edge_list1)
		<type 'numpy.ndarray'>
	>>> edge_list1[:5]
		array([	[0, 1],
       			[0, 2],
       			[1, 2],
       			[0, 3],
       			[1, 3]], dtype=int16)
	'''

	adjacency_mat1, edge_list1 = LoadGraph()
	adjacency_mat2, edge_list2 = LoadGraph()
	
	logging.info('Calling CommuteTimeDistance')
		
	dij1 = CommuteTimeDistances(edge_list1, adjacency_mat1, tol, epsilon, d)	
	dij2 = CommuteTimeDistances(edge_list2, adjacency_mat2, tol, epsilon, d)
	
	#Set this threshold value.
	threshold = 10
	deltaE = GetdeltaE(adjacency_mat1,edge_list1,dij1,adjacency_mat2,edge_list2,dij2)