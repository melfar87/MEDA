from enum import IntEnum
import numpy as np
from stable_baselines.common.schedules import Schedule
import math
import matplotlib.pyplot as plt

class State(IntEnum):
    UNDEFINED = 0
    INIT = 1
    READY = 2
    BUSY = 3
    WAIT = 4
    DONE = 5


class MoTypes(IntEnum):
    UNDEFINED = 0
    DIS = 1
    OUT = 2
    DSC = 3
    MIX = 4
    DLT = 5
    MAG = 6
    WAS = 7
    THM = 8
    SPT = 9
    
    
class Droplet():
    """ Biochip Droplet Class """
    droplet_count = 0
    
    @classmethod
    def reset(cls):
        """ Resets class """
        cls.droplet_count = 0
        return
    
    
    def __init__(self, center=None, size=None, dr_array=None, visible=False):
        if center is not None and size is not None:
            self.center = np.array([center[0], center[1]])
            self.size = np.array([size[0],size[1]])
            self.dr = np.array([center[0]-size[0]/2, center[1]-size[1]/2,
                           center[0]+size[0]/2, center[1]+size[1]/2])
        elif dr_array is not None:
            self.dr = np.array([dr_array[0], dr_array[1],
                                dr_array[2], dr_array[3]])
            self.center = np.array([(dr_array[2]-dr_array[0])/2+dr_array[0],
                                    (dr_array[3]-dr_array[1])/2+dr_array[1]])
            self.size = np.array([(dr_array[2]-dr_array[0]),
                                  (dr_array[3]-dr_array[1])])
        else:
            raise Exception("ERROR: No droplet configuration provided")
        self.visible = visible
        self.id = Droplet.droplet_count
        Droplet.droplet_count += 1
        return
    
    
def showIsGPU():
    import tensorflow as tf
    if tf.test.is_gpu_available():
        print("\n\n\n##### Training on GPUs... #####\n")
    else:
        print("\n\n\n##### Training on CPUs... #####\n")
    return


def showIsGPU(tf):
    if tf.test.is_gpu_available():
        print("\n\n\n##### Training on GPUs... #####\n")
    else:
        print("\n\n\n##### Training on CPUs... #####\n")
    return


class LearningRateSchedule(Schedule):
    def __init__(self, base_rate=2.5e-4):
        self.base_rate:float = float(base_rate)
        return
    
    def value(self, frac):
        return self.linearDecayValue(frac)
    
    def linearDecayValue(self, frac):
        lr_now = math.sqrt(float(frac)) * self.base_rate
        return lr_now


def plotProbVsCycles(k_list, str_filename, str_title='Title'):
    data = np.array(k_list)
    # o_line = np.average(o_rewards, axis = 0)
    a_max = np.max(data, axis = 0) + 1
    a_min = np.min(data, axis = 0) - 1
    # o_max = np.max(o_rewards, axis = 0)
    # o_min = np.min(o_rewards, axis = 0)
    x_data = np.array([i for i in range(a_min,a_max+1)])
    sample_count = data.size
    y_data = np.zeros(x_data.shape)
    for i in range(x_data.size):
        y_data[i] = np.sum(data <= x_data[i])/sample_count

    with plt.style.context('seaborn-paper'):
        plt.rcParams.update({'font.size': 10, 'figure.figsize': (6,4)})
        plt.figure()
        # plt.fill_between(episodes, a_max, a_min, facecolor = 'red', alpha = 0.3)
        # plt.fill_between(episodes, o_max, o_min, facecolor = 'blue',
        #         alpha = 0.3)
        plt.plot(x_data, y_data, 'r-', label = 'DRL')
        # plt.plot(episodes, a_goals_line, 'g.', label = 'Success Rate')
        # plt.plot(episodes, a_cycles_line, 'k-.', label = 'No. Cycles')
        # plt.plot(episodes, o_line, 'b-', label = 'Baseline')
        # if b_path:
            # leg = plt.legend(loc = 'upper left', shadow = True, fancybox = True)
        # else:
        # leg = plt.legend(loc = 'lower right', shadow = True, fancybox = True)
        # leg.get_frame().set_alpha(0.5)
        plt.title(str_title)
        plt.xlabel('Number of Cycles')
        plt.ylabel('Prob. of Success')
        plt.tight_layout()
        # Save PNG
        plt.savefig('figs/' + str_filename + '.png')
        # Save TEX
        import tikzplotlib
        tikzplotlib.clean_figure()
        tikzplotlib.save('figs/' + str_filename + '.tex')
        
    return