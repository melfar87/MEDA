import timeit
import numpy as np
import pickle
from enum import IntEnum
import matplotlib.pyplot as plt




def main(args):
    
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    
    freq_inits:np.array = np.asarray(data.get('debug_info').get('freq_inits'))
    freq_goals:np.array = np.asarray(data.get('debug_info').get('freq_goals'))
    
    freq_init = freq_inits.sum(0)
    freq_goal = freq_goals.sum(0)
    
    with plt.style.context('seaborn-paper'):
        plt.rcParams.update({'font.size': 10, 'figure.figsize': (6,4)})
        
        plt.figure()
        plt.matshow(freq_init)
        plt.savefig('figs/_plot_init2.png')
        
        plt.figure()
        plt.matshow(freq_goal)
        plt.savefig('figs/_plot_goal2.png')
    return

class Args(object):
    pass

if __name__ == '__main__':
    args = Args()
    # args.filename = 'data/0823a_030x030_E100_NPS64.pickle'
    args.filename = 'data/0826a_030x030_E100_NPS64.pickle'
    main(args)






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

# height = 5
# width = 8
# deg_perc = 0.2
# deg_size=1
# A = np.zeros((height,width))
# loc = np.random.choice(A.size, int(height*width*deg_perc/deg_size), replace=False)

# for x in loc:
#     r = x // width
#     c = x % width
#     print("i%d r%d c%d" % (x,r,c))
    
# print(" ")
# # np.put()

# np.random.choice()

# np.indices((3,4))