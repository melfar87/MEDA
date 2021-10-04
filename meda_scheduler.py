from cv2 import DrawMatchesFlags_DRAW_OVER_OUTIMG
from matplotlib.pyplot import tick_params
from envs.meda import MEDAEnv
import numpy as np
from typing import List
from envs.meda import Direction
from meda_utils import *
from meda_biochip import MedaBiochip




class Mo():
    """ Microfluidic Operation Class """
    mo_count = 0
    def __init__(self, s_name, s_type, s_pre, dr_start, dr_goal, cond=None) -> None:
        # super.__init__()
        self.id = Mo.mo_count
        Mo.mo_count += 1
        # [TODO] Fix how lists are dealt with
        self.s_name = s_name
        self.s_type = s_type
        self.s_pre = s_pre
        # self.droplets = []
        # self.droplets.append(dr_start)
        self.droplets = dr_start
        # self.locations = []
        # self.locations.append(dr_goal)
        self.locations = dr_goal
        # self.cond = []
        # self.cond.append(cond)
        self.cond = cond
        self.state = State.UNDEFINED
        self.k = -1
        self.time = -1
        self.rj_list:list[Rj] = []
        return
    
    
    def createRoutingJobs(self) -> None:
        """ Create routing jobs for the current MO """
        # [TODO] Implement Mo.createRoutingJobs()
        if self.s_type in {'Dis'}: # Dispense
            # No job required
            rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0], self)
            self.rj_list.append(rjObj)
        elif self.s_type in {'Out','Dsc'}:
            rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0], self)
            self.rj_list.append(rjObj)
        elif self.s_type in {'Mix','Dlt'}: # Mixing or Diluting
            rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0], self)
            self.rj_list.append(rjObj)
            rjObj:Rj = Rj('J1', self.droplets[1], self.locations[0], self)
            self.rj_list.append(rjObj)
        elif self.s_type in {'Mag','Was','Thm'}:
            rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0], self)
            self.rj_list.append(rjObj)
        elif self.s_type in {'Spt'}: # Split
            rjObj:Rj = Rj('J0', self.droplets[0], self.locations[0], self)
            self.rj_list.append(rjObj)
            rjObj:Rj = Rj('J1', self.droplets[1], self.locations[0], self)
            self.rj_list.append(rjObj)
        else:
            raise Exception("Unknown type: %s" % self.s_type)
        self.state = State.READY
        return




class Rj():
    """ Routing Job Class """
    rj_count = 0
    def __init__(self, s_name, dr_start, dr_goal, parent=None) -> None:
        # super.__init__()
        self.id = Rj.rj_count
        Rj.rj_count += 1
        self.s_name = s_name
        self.dr_droplet = np.array(dr_start)
        self.dr_goal = np.array(dr_goal)
        self.parent = parent
        self.hazard = np.zeros(4, dtype=np.int)
        self.b_show = False
        self.env_id = None
        self.obs = None
        self.state:State = State.INIT
        self.b_is_done = False
        self.k = -1
        self.time = -1
        # self.shift = None
        self.tra_mirror_x = None
        self.tra_mirror_y = None
        self.tra_delta_x = None
        self.tra_delta_y = None
        self.tra_delta = None
        self.tra_mirror = None
        self.env_droplet = None
        self.env_goal = None
        return
    
    
    def isDone(self):
        # b_done = np.array_equal(self.dr_droplet, self.dr_goal)
        b_done = self.b_is_done
        return b_done
    
    



class MedaScheduler():
    """ MEDA Biochip Scheduler """
    
    def __init__(self, biochip:MedaBiochip, env=None, policy=None, width=0, height=0):
        """ Initialize scheduler """
        # super(MedaScheduler, self).__init__()
        self.biochip:MedaBiochip = biochip
        self.env_list = env.envs
        self.m_pattern:np.ndarray = np.zeros_like(env.envs[0].m_pattern)
        self.m_actcount:np.ndarray = np.zeros_like(env.envs[0].m_actcount)
        self.env_rj_ids:List = [None] * env.num_envs
        self.policy = policy
        self.width = width
        self.height = height
        self.policy = policy
        self.env = env
        self.obs = self.env.reset()
        self.ticks = 0 # clock
        self.state = State.INIT # scheduler state
        self.mo_list:list[Mo] = []
        self.rj_list:list[Rj] = []
        self.dr_list:list[Droplet] = []
        self.mo_dict = {}
        self.dis_list:list[Rj] = [] # list of busy dispensing jobs
        self.ticks = 0
        return
    
    
    def reset(self):
        self.obs = self.env.reset()
        self.m_actcount = np.random.randint(0,high=400,size=self.m_actcount.shape)
        self.m_pattern.fill(0)
        # self.m_actcount.fill(0)
        moObj:Mo
        for moObj in self.mo_list:
            moObj.state = State.READY
            moObj.k = -1
            moObj.time = -1   
        rjObj:Rj
        for rjObj in self.rj_list:
            rjObj.state = State.INIT
            rjObj.k = -1
            rjObj.time = -1
            rjObj.b_is_done = False
            rjObj.b_show = False
        self.dis_list = []
        self.ticks = 0
        self.state = State.INIT
        return
    
    
    def addMo(self, s_name, s_type, s_pre, dr_start, dr_end, cond=None):
        """ Add microfluidic operation """
        moObj = Mo(s_name,s_type,s_pre,dr_start,dr_end,cond)
        self.mo_list.append(moObj)
        self.mo_dict[moObj.s_name] = moObj
        return
    
    
    def preprocessAllMos(self) -> None:
        """ Preprocess all MOs and generate list of routing jobs """
        moObj:Mo
        for moObj in self.mo_list:
            moObj.createRoutingJobs()
            rjObj:Rj
            for rjObj in moObj.rj_list:
                self.rj_list.append(rjObj)
                drObj = Droplet(dr_array=rjObj.dr_droplet, visible=False)
                self.dr_list.append(drObj)
        self.biochip.setDroplets(self.dr_list)
        return
    
    
    def tick(self):
        """ Run one control loop """
        self.droplets = self.biochip.getDroplets()
        self.processStates()
        # self.env.envs[0].render(mode='human_frame') # Render
        self.updateControlActions()
        # self.env.envs[0].render(mode='human_frame') # Render
        # self.applyControlActions()
        self.biochip.setDroplets(self.droplets)
        self.ticks += 1
        return self.state
    
    
    def applyControlActions(self):
        """ Applies control actions for all RJs """
        # rjObj:Rj
        # for rjObj in self.rj_list:
            
        raise NotImplementedError
    
    
    def processStates(self):
        """ Process scheduler state machine """
        b_repeat = True
        while b_repeat:
            b_repeat = False
            moObj:Mo # typing
            for moObj in self.mo_list:
                # INIT State
                if moObj.state == State.INIT:
                    print("WARNING: MO %s was not processed" % moObj.s_name)
                    pass
                # READY State
                elif moObj.state == State.READY:
                    b_can_start = True
                    for preObjName in moObj.s_pre:
                        preObj:Mo = self.mo_dict[preObjName]
                        b_can_start = b_can_start and preObj.state == State.DONE
                    for condObjName in moObj.cond:
                        condObj:Mo = self.mo_dict[condObjName]
                        b_can_start = b_can_start and condObj.state == State.DONE
                    if b_can_start:
                        moObj.k = 0
                        moObj.time = self.ticks
                        rjObj:Rj # typing
                        for rjObj in moObj.rj_list:
                            # [TODO] Strategy should be synthesized here
                            # [TODO] Mark current rjObj as visible
                            # self.JobStates[moId][jobId] = [2,jobObj.droplet]
                            self._assignEnv(rjObj)
                            rjObj.b_show = True
                            rjObj.time = self.ticks
                            rjObj.k = 0
                        for preObjName in moObj.s_pre:
                            preObj:Mo = self.mo_dict[preObjName]
                            preRjObj:Rj # typing
                            for preRjObj in preObj.rj_list:
                                preRjObj.b_show = False
                        moObj.state = State.BUSY
                        b_repeat = True
                # BUSY State
                elif moObj.state == State.BUSY:
                    b_is_done = True
                    for rjObj in moObj.rj_list:
                        b_is_done = b_is_done and rjObj.isDone()
                    if b_is_done:
                        moObj.state = State.DONE
                        if moObj.s_type in {'Out','Dsc'}:
                            moObj.rj_list[0].b_show = False
                        # [NOTE] Release moved to fix multi-rj MOs
                        # for rjObj in moObj.rj_list:
                        #    self._releaseEnv(rjObj)
                        b_repeat = True
                    moObj.k += 1
                # DONE State
                elif moObj.state == State.DONE:
                    pass # do nothing
                # Unknown
                else:
                    print("ERROR: Unknown state %s" % moObj.state.name)
        # Update internal state
        if all([moObj.state==State.DONE for moObj in self.mo_list]):
            self.state = State.DONE
        elif any([moObj.state==State.BUSY for moObj in self.mo_list]):
            self.state = State.BUSY
        else:
            self.state = State.READY
        return
    
    
    def isDone(self) -> bool:
        """ Returns True if bioassay execution ended """
        results = (self.state == State.DONE)
        return results
    
    
    def updateControlActions(self):
        """ Updates control actions for all RJs """
        # control pattern
        self.m_pattern.fill(0)
        # [NOTE][2021-07-25] Added deterministic=True parameter
        action, state = self.policy.predict(self.obs, deterministic=True)
        self.obs, reward, done, _info = self.env.step(action)
        
        for env_id, rj_id in enumerate(self.env_rj_ids):
            if rj_id is not None:
                self.rj_list[rj_id].b_is_done = done[env_id]
                self.rj_list[rj_id].k += 1
                if self.rj_list[rj_id].b_is_done:
                    self._releaseEnv(self.rj_list[rj_id])
                self.m_pattern += self.env.envs[env_id].prev_m_pattern
                
        
        # print(done)
        # print(reward)
        # print(action)
        
        # Process dispensing operations separately
        rjObj:Rj
        for rjObj in self.dis_list:
            if not all(rjObj.dr_goal==rjObj.dr_droplet):
                # perform action
                if rjObj.dr_goal[0]!=rjObj.dr_droplet[0]:
                    diff = rjObj.dr_goal[0] - rjObj.dr_droplet[0]
                    step = (rjObj.dr_droplet[2] - rjObj.dr_droplet[2])*(diff//abs(diff))
                    shift = min(diff,step//2,key=abs)
                    rjObj.dr_droplet += [shift,0,shift,0]
                elif rjObj.dr_goal[1]!=rjObj.dr_droplet[1]:
                    diff = rjObj.dr_goal[1] - rjObj.dr_droplet[1]
                    step = (rjObj.dr_droplet[3] - rjObj.dr_droplet[1])*(diff//abs(diff))
                    shift = min(diff,step//2,key=abs)
                    rjObj.dr_droplet += [0,shift,0,shift]
                else:
                    raise Exception("Something is wrong - aborting...")
                
                if all(rjObj.dr_goal==rjObj.dr_droplet):
                    rjObj.b_is_done = True
                
            else:
                rjObj.b_is_done = True
        
        
        np.clip(self.m_pattern,0,1,out=self.m_pattern)
        self.m_actcount += self.m_pattern
        # [FIXME] Remove dispensing jobs that are finished from self.dis_list
        
        # moObj:Mo # typing
        # for moObj in self.mo_list:
        #     if moObj.state == State.BUSY:
        #         rjObj:Rj # typing
        #         for rjObj in moObj.rj_list:
        #             # [TODO] Set actions
        #             pass
        #     elif moObj.state == State.DONE:
        #         rjObj:Rj # typing
        #         for rjObj in moObj.rj_list:
        #             # [TODO] Set actions
        #             pass
        #     else:
        #         rjObj:Rj # typing
        #         for rjObj in moObj.rj_list:
        #             # [TODO] Set actions
        #             pass
        return
    
    
    def _assignEnv(self, rjObj:Rj):
        """ Assigns and initializes the next available environment to RJ """
        # Skip assignment if it is a dispensing operation
        if rjObj.parent.s_type == 'Dis':
            self.dis_list.append(rjObj)
            return
        
        # Everything else: assign an evnironment
        env_idx = next(
            (i for i,v in enumerate(self.env_rj_ids) if v is None), None)
        if env_idx is None: raise Exception("Out of Enviromnets - Terminating")
        self.env_rj_ids[env_idx] = rjObj.id
        rjObj.env_id = env_idx
        rj_env:MEDAEnv = self.env_list[env_idx]
        # obs = env.reset()
        
        # if rjObj.dr_droplet[0] > rjObj.dr_goal[0]:
        #     rjObj.tra_mirror_x = True
        #     rjObj.tra_delta_x = self.width - rjObj.dr_droplet[2] - 3
        # else:
        #     rjObj.tra_mirror_x = False
        #     rjObj.tra_delta_x = 0 - rjObj.dr_droplet[0] + 3
        
        # if rjObj.dr_droplet[1] > rjObj.dr_goal[1]:
        #     rjObj.tra_mirror_y = True
        #     rjObj.tra_delta_y = self.height - rjObj.dr_droplet[3] - 3
        # else:
        #     rjObj.tra_mirror_y = False
        #     rjObj.tra_delta_y = 0 - rjObj.dr_droplet[1] + 3
            
        # rjObj.tra_delta = np.array([rjObj.tra_delta_x, rjObj.tra_delta_y,
        #                             rjObj.tra_delta_x, rjObj.tra_delta_y])
        
        # rjObj.env_droplet = self._rjToEnv(rjObj, rjObj.dr_droplet)
        # rjObj.env_goal = self._rjToEnv(rjObj, rjObj.dr_goal)
        
        # rjObj.hazard[:] = (
        #     max(min(rjObj.dr_goal[0], rjObj.dr_droplet[0])-3,0),
        #     max(min(rjObj.dr_goal[1], rjObj.dr_droplet[1])-3,0),
        #     min(max(rjObj.dr_goal[2], rjObj.dr_droplet[2])+3,self.width),
        #     min(max(rjObj.dr_goal[3], rjObj.dr_droplet[3])+3,self.height)
        # )
        
        # [NOTE][2021-07-25] This replaced the above code
        rjObj.tra_delta_x = \
            max(0, 0-rjObj.dr_droplet[0], 0-rjObj.dr_goal[0]) - \
            max(0, rjObj.dr_droplet[2]-self.width, rjObj.dr_goal[2]-self.width)
        rjObj.tra_delta_y = \
            max(0, 0-rjObj.dr_droplet[1], 0-rjObj.dr_goal[1]) - \
            max(0, rjObj.dr_droplet[3]-self.height, rjObj.dr_goal[3]-self.height)
        rjObj.tra_delta = np.array([rjObj.tra_delta_x, rjObj.tra_delta_y,
                                    rjObj.tra_delta_x, rjObj.tra_delta_y])
        
        rjObj.env_droplet = self._rjToEnv(rjObj, rjObj.dr_droplet)
        rjObj.env_goal = self._rjToEnv(rjObj, rjObj.dr_goal)
        
        
        # Provide the parameters relative to the routing job
        # [TODO][CRITICAL] Provide correct setState params
        
        rjObj.obs = rj_env.setState(
            dr_s=rjObj.env_droplet,
            dr_g=rjObj.env_goal,
            m_actcount=self.m_actcount
        )
        
        self.obs[rjObj.env_id] = rjObj.obs
        return
    

    def _releaseEnv(self, rjObj:Rj):
        if rjObj.parent.s_type == 'Dis': return
        self.env_rj_ids[rjObj.env_id] = None
        rjObj.env_id = None
        return
    
    
    def _bioToEnv(self, rjObj:Rj):
        pass


    def _rjToEnv(self, rjObj:Rj, dr, b_transform=False):
        if b_transform:
            dr_env = dr + rjObj.tra_delta
            if rjObj.tra_mirror_x:
                dr_env[0] = self.width - dr_env[0]
                dr_env[2] = self.width - dr_env[2]
            if rjObj.tra_mirror_y:
                dr_env[1] = self.height - dr_env[1]
                dr_env[3] = self.height - dr_env[3]
        else:
            dr_env = dr
        return dr_env
            

    # def preprocessAllRjs(self) -> None:
    #     """ Preprocess all RJs """
    #     # [TODO] Implement processAllRjs
    #     moObj:Mo # typing
    #     for moObj in self.mo_list:
            
    #         rjObj:Rj
    #         for rjObj in moObj.rj_list:
    #             self.rj_list.append(rjObj)
    #     raise NotImplementedError


