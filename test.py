import os, subprocess

ps = {}
cmd = ["/usr/local/src/spark-2.0.0-bin-hadoop2.7/bin/spark-submit", "/Users/Pragya/Documents/SDL/SLD-C1/CAD.py", "/Users/Pragya/Documents/SDL/SLD-C1/options.json"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
# ps[p.pid] = p
out, err = p.communicate()
print out
# p = subprocess.call(cmd, shell = True)
# print "Waiting for %d processes..." % len(ps)
# while ps:
#     pid, status = os.wait()
#     if pid in ps:
#         del ps[pid]
#         print "Waiting for %d processes..." % len(ps)

print "XYZ"