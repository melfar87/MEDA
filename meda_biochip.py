from meda_utils import Droplet
import numpy as np



class MedaBiochip():
    """ MEDA Biochip Class """
    def __init__(self, env=None, policy=None, width=0, height=0) -> None:
        self.env_list = env.envs
        self.policy = policy
        self.height = height
        self.width = width
        self.m_degradation = np.ones((width, height))
        self.m_health = np.ones((width, height))
        # self.m_actcount = np.zeros((width, height))
        self.m_pattern = np.zeros((width, height), dtype=np.uint8)
        self.m_taus =np.ones_like(env.envs[0].m_taus)*0.7
        self.m_c1s = np.zeros_like(env.envs[0].m_C1s)
        self.m_c2s = np.ones_like(env.envs[0].m_C2s)*300
        self.m_actcount = np.zeros_like(env.envs[0].m_actcount)
        
        # Masking matrices used for shifting purposes
        self.m_zeros = np.zeros((width*3, height*3), dtype=np.uint8)
        self.m_ones  = np.ones((width*3, height*3), dtype=np.uint8)
        
        self.dr_list:[Droplet] = []
        
        # raise NotImplementedError
    
    
    def getDroplets(self):
        dr_list = self.dr_list
        return dr_list
    
    
    def setDroplets(self, dr_list):
        self.dr_list = dr_list
        # [TODO] Update view based on new list
        return