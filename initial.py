import scipy.io, numpy as np
from GraphFactory import *
import subprocess

# A little function to map year and month to index in matrix.
def convertDateToIndex(year, month):
	if year < 1982 or year > 2002:
		raise ValueError('Year not supported. Data available only for years between(inclusive) 1982 and 2002')
	if month < 0 or month > 12 or type(month) != int:
		raise ValueError('Unregonized month number. Please provide valid month number (1-12)')
	return (year - 1982) * 12 + month - 1 # Subtract by 1 to account for python range

x = scipy.io.loadmat('/home/ldapuser1/code-from-git/SLD-C1/temColFormat.mat')
matrix = np.array(x['tem'])
jan1994Index = convertDateToIndex(1994,1) ## January, 1994
ndarray1 = [[0 if e[jan1994Index] == -9999 else e[jan1994Index] for e in row] \
	for row in matrix]

jan1995Index = convertDateToIndex(1995,1) ## January, 1995
ndarray2 = [[0 if e[jan1995Index] == -9999 else e[jan1995Index] for e in row] \
	for row in matrix]

sig = 1/(2**0.5)
graphFactory = GraphFactory(ndarray1, sig, 4, '/home/ldapuser1/code-from-git/SLD-C1/January1994/', False)
graphFactory.createGraph()
graphFactory2 = GraphFactory(ndarray2, sig, 4, '/home/ldapuser1/code-from-git/SLD-C1/January1995/', False)
graphFactory2.createGraph()
print "Starting CAD"
cmd = ["/home/ldapuser1/spark-2.0.2-bin-hadoop2.4/bin/spark-submit", "/home/ldapuser1/code-from-git/SLD-C1/CAD.py", "/home/ldapuser1/code-from-git/SLD-C1/optionsTest.json"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
output, error = proc.communicate()
print output
# x, y, spacing, iterations, learningRate, initialSigma):
logfname = 'LE_'+datetime.now().strftime('%Y-%m-%d-%H:%M:%S')+'.log'
logging.basicConfig(filename=logfname,filemode='w',level= logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')	
le = LearningEngine('/home/ldapuser1/code-from-git/SLD-C1/le-config.json', logging)
le.runForOneRate(x, y, 4, 20, 0.5, 1)

