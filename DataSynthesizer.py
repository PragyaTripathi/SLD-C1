import numpy as np
from random import randint
from GraphFactory import *
import matplotlib.pyplot as plt

def sytheticDATA(size): 
	mean1 = [5, 5]
	cov1 = [[3, 0], [0, 1.5]]
	x1, y1 = np.random.multivariate_normal(mean1, cov1, size).T

	mean2 = [18, 15]
	cov2 = [[1, 0], [0, 2]]
	x2, y2 = np.random.multivariate_normal(mean2, cov2, size).T

	mean3 = [9, 10]
	cov3 = [[3, 0], [0, 2]]
	x3, y3 = np.random.multivariate_normal(mean3, cov3, size).T

	mean4 = [20, 8]
	cov4 = [[2, 0], [0, 3]]
	x4, y4 = np.random.multivariate_normal(mean4, cov4, size).T

	# plt.plot(x1, y1,'+',x2, y2,'+', x3, y3,'+',x4,y4,'+')
	# plt.show()

	data1_x = np.concatenate((x1, x2, x3, x4), axis=0)
	data1_y = np.concatenate((y1, y2, y3, y4), axis=0)
	index = range(4*size)
	data1 = np.column_stack((data1_x,data1_y))
	# plt.plot(data1_x, data1_y,'o')
	# add white noise
	mean_whitenoise = [0, 0]
	cov_whitenoise = [[0.8, 0.8], [0.8, 0.8]]
	whitenoise_x, whitenoise_y = np.random.multivariate_normal(mean_whitenoise, cov_whitenoise, 4*size).T
	data2_x = np.add(data1_x, whitenoise_x)
	data2_y = np.add(data1_y, whitenoise_y)
	# plt.plot(data2_x, data2_y,'+')

	#sort data1 based on norm val
	norm_data1 = [np.linalg.norm(data1[i]) for i in range(size*4)]
	index = range(4*size)
	new_norm_data1 = np.column_stack((norm_data1,index))
	sort_array = new_norm_data1[new_norm_data1[:,0].argsort()][::-1]
	
	# generate anomolous nodes
	anomoulous_index = sort_array[:size*4/10][:, 1]
	print(anomoulous_index)
	mean_noise = [5, 5]
	cov_noise = [[0.8, 0.8], [0.8, 0.8]]
	noise_x, noise_y = np.random.multivariate_normal(mean_noise, cov_noise, anomoulous_index.size).T
	for i in range(anomoulous_index.size):
		data2_x[anomoulous_index[i]] += noise_x[i]
		data2_y[anomoulous_index[i]] += noise_y[i]
	# print data2_x, data2_y
	# plt.plot(data2_x, data2_y,'.')
	# plt.show()
	return np.column_stack((data1_x,data1_y)), np.column_stack((data2_x,data2_y))

# x, y = sytheticDATA(16)
# # # print x
# # # print x.shape
# # sigma = 1/(2**0.5)
# graphFactory = GraphFactory(x, sigma, 0, '/Users/Pragya/Documents/SDL/SLD-C1/SyntheticDataTest/', True)
# graphFactory.createGraph()
# graphFactory2 = GraphFactory(y, sigma, 0, '/Users/Pragya/Documents/SDL/SLD-C1/SyntheticDataWithAnomaliesTest/', True)
# graphFactory2.createGraph()
# print "Starting CAD"
# cmd = ["/usr/local/src/spark-2.0.0-bin-hadoop2.7/bin/spark-submit", "/Users/Pragya/Documents/SDL/SLD-C1/CAD.py", "/Users/Pragya/Documents/SDL/SLD-C1/optionsTest.json"]
# p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
# out, err = p.communicate()
# print out