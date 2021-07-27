import timeit
import numpy as np
import pickle
from enum import IntEnum


# filename = "MDL_A_120x120_E051_NS30V300.tex"
# with open("./log/" + filename, "rt") as fin:
#     with open("./log2/" + filename, "wt") as fout:
#         for line in fin:
#             fout.write(line.replace('line width=0.56pt', 'line width=1.00pt'))


# with open("./data/TMP_030x030_E002_FIX.pickle", 'rb') as f:
#     data_obj = pickle.load(f)
    
# print(data_obj)


# y = np.zeros([100,100]) + 1
# x = np.array([100,100])
# x.fill(0.5)
# t = timeit.Timer("x/y.sum()", "from __main__ import x,y")
# time = t.timeit(10000)
# print("Time %f\n" % (time,))

# y = np.zeros([100,100]) + 1
# x = np.array([100,100])
 
# x = 'some_str'
# t = timeit.Timer("y = x=='some_strs'", "from __main__ import x")
# time = t.timeit(10000)
# print("Time %f\n" % (time,))

height = 5
width = 8
deg_perc = 0.2
deg_size=1
A = np.zeros((height,width))
loc = np.random.choice(A.size, int(height*width*deg_perc/deg_size), replace=False)

for x in loc:
    r = x // width
    c = x % width
    print("i%d r%d c%d" % (x,r,c))
    
print(" ")
# np.put()

np.random.choice()

np.indices((3,4))