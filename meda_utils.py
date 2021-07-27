from enum import IntEnum
import numpy as np



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

# class Biochip():
#     """ MEDA Biochip State Class """
#     def __init__(self) -> None:
#         pass