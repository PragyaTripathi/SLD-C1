import scipy.io, numpy as np
from GraphFactory import *

x = scipy.io.loadmat('/Users/Pragya/Documents/SDL/SLD-C1/temColFormat.mat')
jan1994Index = convertDateToIndex(1996,1) ## January, 1994
matrix = np.array(x['tem'])
ndarray = [[0 if e[jan1994Index] == -9999 else e[jan1994Index] for e in row] \
	for row in matrix]
graphFactory = GraphFactory(ndarray, 1, 4, '/Users/Pragya/Documents/SDL/SLD-C1/January1996/')
graphFactory.createGraph()

