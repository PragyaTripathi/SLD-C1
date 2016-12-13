import os, subprocess

ps = {}
cmd = ["/home/ldapuser1/spark-2.0.2-bin-hadoop2.4/bin/spark-submit", "/home/ldapuser1/code-from-git/SLD-C1/CAD.py", "/home/ldapuser1/code-from-git/SLD-C1/options.json"]
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