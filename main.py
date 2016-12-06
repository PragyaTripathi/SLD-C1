from DataSynthesizer import *
from GraphFactory import *
from LearningEngine import *

x, y = sytheticDATA(16)
sigma = 1/(2**0.5)
graphFactory = GraphFactory(x, sigma, 0, '/Users/Pragya/Documents/SDL/SLD-C1/SyntheticData2/', True)
graphFactory.createGraph()
graphFactory2 = GraphFactory(y, sigma, 0, '/Users/Pragya/Documents/SDL/SLD-C1/SyntheticDataWithAnomalies2/', True)
graphFactory2.createGraph()
print "Starting CAD"
cmd = ["/usr/local/src/spark-2.0.0-bin-hadoop2.7/bin/spark-submit", "/Users/Pragya/Documents/SDL/SLD-C1/CAD.py", "/Users/Pragya/Documents/SDL/SLD-C1/options.json"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
out, err = p.communicate()
print out
le = LearningEngine('/Users/Pragya/Documents/SDL/SLD-C1/le-config.json')
# le.runForOneRate(x, y, 1, 100, 10)
# le.runForOneRate(x, y, 1, 2, 10)
# le.runForOneRate(x, y, 1, 10, 200)
le.runWithRestarts(x, y, 1, 100, 10, 10)