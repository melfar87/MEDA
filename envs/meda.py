import copy
import random
import numpy as np
from enum import IntEnum
# import matplotlib
# matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

import gym
from gym import error, spaces, utils
# from gym.wrappers import ResizeObservation
import cv2


# class DirectionSimple(IntEnum):
#     NN = 0  # North
#     EE = 1  # East
#     SS = 2  # South
#     WW = 3  # West
    

# class Direction(IntEnum):
#     ZZ = 0  # Sleep/Stop
#     NN = 1  
#     NE = 2  
#     EE = 3  
#     SE = 4  
#     SS = 5  
#     SW = 6  
#     WW = 7  
#     NW = 8
    
class Direction(IntEnum):
    NN = 0 
    NE = 1 
    EE = 2 
    SE = 3 
    SS = 4 
    SW = 5 
    WW = 6 
    NW = 7    



class MEDAEnv(gym.Env):
    """ MEDA biochip environment, following gym interface """
    metadata = {'render.modes': ['human', 'human_frame', 'rgb_array'],
                'video.frames_per_second': 2}

    def __init__(self, width=0, height=0, droplet_sizes=[[4,4],], n_bits=2,
                 b_degrade=True, b_use_dict=False,
                 b_unify_obs=True, b_parm_step=True, obs_size=(30, 30),
                 deg_mode='normal', deg_perc=0.1, deg_size=1,
                 b_play_mode=False, delay_counter = 10, debug=0,
                 sampling="stratified",
                 **kwargs):
        """ Gym Constructor for MEDA
        :param height: Biochip height in microelectrodes
        :param width: Biochip width in microelectrodes
        :param b_random:
        :param n_modules:
        :param b_degrade:
        :param per_degrade:
        :param b_use_dict: Use dictionary for observation space if true
        """
        super(MEDAEnv, self).__init__()
        self.viewer = None
        self.debug = debug
        self.actions = Direction
        
        # Play mode
        try:
            self.b_play_mode = kwargs['b_play_mode']
        except:
            self.b_play_mode = b_play_mode
        print("INFO: Play mode is %s" % ("ENABLED" if self.b_play_mode else "DISABLED"))
        self.delay_counter = 0
        self.def_delay_counter = delay_counter
        
        # Configs
        width, height = kwargs['size']
        self.obs_size = obs_size
        self.height = height
        self.width = width
        assert height > 4 and width > 4
        self.n_bits = n_bits
        self.b_unify_obs = b_unify_obs
        self.b_parm_step = b_parm_step
        self.droplet_sizes = droplet_sizes
        self.b_degrade = b_degrade
        self.deg_mode = deg_mode
        self.deg_perc = deg_perc
        self.deg_size = deg_size
        self.sampling = sampling
        self.xs0range = [3,]
        self.ys0range = [1,3,]
        self.act_count = np.zeros(9)
        self.action = Direction(0)
        
        # State vars
        self.droplet = np.zeros(4, dtype=np.int)
        self.goal = np.zeros(4, dtype=np.int)
        self.hazard = np.zeros(4, dtype=np.int)
        self.m_taus = np.ones((width, height)) * 0.7
        self.m_C1s = np.ones((width, height)) * 0
        self.m_C2s = np.ones((width, height)) * 200
        self.m_degradation = np.ones((width, height))
        self.m_health = np.ones((width, height))
        self.m_actcount = np.zeros((width, height))
        self.m_pattern = np.zeros((width, height), dtype=np.uint8)
        self.prev_m_pattern = np.zeros((width, height), dtype=np.uint8)
        self.step_count = 0
        self.max_step = 0
        self.violation_count = 0
        self.collision = np.zeros(4, dtype=np.bool)

        # Gym environment: rewards and action space
        self.reward_range = (-1000.0, 1000.0)
        self.action_space = spaces.Discrete(len(self.actions))
        if b_use_dict:
            # self.observation_space = spaces.Dict({
            #     'health': spaces.Box(
            #         low=0, high=2**n_bits-1,
            #         shape=(width, height, 1), dtype=np.uint8),
            #     'sensors': spaces.Box(
            #         low=0, high=1,
            #         shape=(width, height), dtype=np.uint8)
            # })
            # self.keys = list(self.observation_space.spaces.keys())
            raise NotImplemented("Dictionary-based observation not implemented")
        else:
            self.n_layers = 3
            self.goal_layer_id = 0
            self.droplet_layer_id = 1
            self.health_layer_id = 2
            self.default_observation = np.zeros(
                shape=(width, height, self.n_layers), dtype=np.float)
            if self.b_unify_obs:
                # Use unified observation size
                self.observation_space = spaces.Box(
                    low=0, high=1,
                    shape=(self.obs_size[0], self.obs_size[1], self.n_layers), dtype=np.float)
            else:
                # Use biochip size for observation
                self.observation_space = spaces.Box(
                    low=0, high=1,
                    shape=(width, height, self.n_layers), dtype=np.float)
        
        # Keys mapping for play mode
        try:
            # from pyglet.window import key
            import pygame
            self.keys_to_action = {
                (pygame.K_w, ): Direction.NN, (pygame.K_KP8, ): Direction.NN,
                (pygame.K_x, ): Direction.SS, (pygame.K_KP2, ): Direction.SS,
                (pygame.K_s, ): Direction.SS, (pygame.K_KP5, ): Direction.SS,
                (pygame.K_d, ): Direction.EE, (pygame.K_KP6, ): Direction.EE,
                (pygame.K_a, ): Direction.WW, (pygame.K_KP4, ): Direction.WW,
                (pygame.K_e, ): Direction.NE, (pygame.K_KP9, ): Direction.NE,
                (pygame.K_q, ): Direction.NW, (pygame.K_KP7, ): Direction.NW,
                (pygame.K_c, ): Direction.SE, (pygame.K_KP3, ): Direction.SE,
                (pygame.K_z, ): Direction.SW, (pygame.K_KP1, ): Direction.SW,
            }
        except:
            print("INFO: PYGLET failed to load")
            self.keys_to_action = None
        
        # Sampling distribution
        # if self.sampling=="stratified":
        #     self.x_set = [i for i in range(self.width)]
        #     self.x_set = [i for i in range(self.width)]
        # elif self.sampling=="uniform":
        #     self.x_set = [i for i in range(self.width)]
        #     self.x_set = [i for i in range(self.width)]
        # else:
        #     raise Exception("Unknown sampling option -- aborting!")
        
        # Sampling function
        max_droplet_x, max_droplet_y = max(self.droplet_sizes) # [FIXME]
        if self.sampling=="stratified":
            excess_x = self.width // 5
            excess_y = self.height // 5
            self.x_set = [0]*(excess_x) + [i for i in range(1,self.width-1)] + [self.width-1]*(excess_x)
            self.y_set = [0]*(excess_y) + [i for i in range(1,self.width-1)] + [self.width-1]*(excess_y) 
            self.dr_size_set = copy.deepcopy(self.droplet_sizes)*10
            self.x_set_buff, self.y_set_buff, self.dr_size_set_buff = [], [], []
            self.fn_get_sample = self._getStratifiedSample
        elif self.sampling=="uniform":
            self.fn_get_sample = self._getUniformSample
        else:
            raise Exception("Unknown sampling option -- aborting!")
            
        # Reset initial state
        self._resetActuationMatrix()
        self._resetInitialState()
        self._resetInitialHealth()
        self._updateHealth()
        
        if self.debug:
            self.freq_goal = np.zeros((width, height), dtype=np.int)
            self.freq_init = np.zeros((width, height), dtype=np.int)
            self.dr_path = []
            self.reset_count = 0
        return

    
    def get_keys_to_action(self):
        return self.keys_to_action

    def printInfo(self):
        print("%s\n%s\n%s" % (self.goal,self.droplet,self.hazard))
        return

    # Gym Interfaces
    def reset(self):
        """Reset environment state 

        Returns:
            obs: Observation
        """
        self.step_count = 0
        self.violation_count = 0
        self._resetActuationMatrix(is_random=False)
        self._resetInitialState()
        self._resetInitialHealth()
        self._updateHealth()
        obs = self._getObs()
        if self.debug:
            self.dr_path = [copy.deepcopy(self.droplet),] # For debugging only
            x0,y0,x1,y1 = self.droplet
            self.freq_init[x0:x1,y0:y1] += 1
            x0,y0,x1,y1 = self.goal
            self.freq_goal[x0:x1,y0:y1] += 1
            self.reset_count +=1
        return obs
    
    
    def setState(self, dr_s, dr_g,
                 m_actcount=None, m_taus=None, m_c1s=None, m_c2s=None):
        """Set biochip state

        Returns:
            obs: Observation
        """
        self.step_count = 0
        # [FIXME] Review what needs to be reset
        
        self.droplet[:] = dr_s[:]
        self.goal[:] = dr_g[:]
        if m_taus is not None: self.m_taus = copy.deepcopy(m_taus)
        if m_c1s is not None: self.m_C1s = copy.deepcopy(m_c1s)
        if m_c2s is not None: self.m_C2s = copy.deepcopy(m_c2s)
        if m_actcount is not None: self.m_actcount = copy.deepcopy(m_actcount)
        
        # Compute hazard bounds
        self.hazard[:] = (
            max(min(self.goal[0],self.droplet[0])-3,0),
            max(min(self.goal[1],self.droplet[1])-3,0),
            min(max(self.goal[2],self.droplet[2])+3,self.width),
            min(max(self.goal[3],self.droplet[3])+3,self.height)
        )
        # Set the max number of steps allowed
        self.max_step = 1*(self.hazard[2]-self.hazard[0]+self.hazard[3]-self.hazard[1])
        # self.max_step = 2*(self.width+self.height)
        
        self._updateHealth()
        obs = self._getObs()
        return obs
    
    
    def step(self, action):
        """Execute one step

        Args:
            action (enum 'Direction'): Action to execute

        Returns:
            [type]: Returns (
                obeservation,
                reward,
                done flag,
                dictionary
                )
        """
        self.act_count[action] += 1
        self.action = Direction(action)
        
        # Keyboard debouncing in play mode
        if self.b_play_mode:
            if self.delay_counter==0:
                if action != 0:
                    # Accept action and start delay counter
                    self.delay_counter = self.def_delay_counter
            elif self.delay_counter > 0:
                if action != 0:
                    # Reject action and keep counting down
                    action = 0
                else:
                    # Action released, so stop counting down
                    self.delay_counter = 0
            else:
                print("WTF")
            
            # Ignore "loiter" action in play mode 
            if action !=0:
                obs = self._getObs()
                reward = 0
                done = self._isComplete() or (self.step_count > self.max_step)
                b_at_goal = 100 if done else 0
                return obs, reward, done, {
                    "b_at_goal":b_at_goal, "num_cycles":self.step_count}

        self.step_count += 1
        prev_dist = self._getDistanceToGoal()
        b_is_valid = self._updatePattern(action)
        if b_is_valid==False: self.violation_count += 1 
        if self.debug:
            self.dr_path.append(copy.deepcopy(self.droplet))
        curr_dist = self._getDistanceToGoal()
        self.m_actcount += self.m_pattern
        self._updateHealth()
        obs = self._getObs()
        
        # reward, b_at_goal, done = self._getRewardA(
        #     b_is_valid=b_is_valid, prev_dist=prev_dist, curr_dist=curr_dist)
        reward, b_at_goal, done = self._getRewardH(
            b_is_valid=b_is_valid, prev_dist=prev_dist, curr_dist=curr_dist)
            
        return obs, reward, done, {"b_at_goal":b_at_goal, "num_cycles":self.step_count}


    def _getRewardH(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        reward = 0.0
        abs_dist = curr_dist
        if self._isComplete():
            reward += 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        else:
            # reward += 1.0/abs_dist**2 - 0.2
            reward = 0.0
            
        if curr_dist < prev_dist:
            reward += 0.5*(prev_dist-curr_dist)
        else:
            reward += 0.8*(prev_dist-curr_dist) - 1.0
            
        if self.step_count > self.max_step:
            reward += 0 
            done = True
            
        if b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward += -1.0
            
        return reward, b_at_goal, done

    def _getRewardG(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        reward = 0.0
        abs_dist = curr_dist
        if self._isComplete():
            reward += 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        else:
            # reward += 1.0/abs_dist**2 - 0.2
            reward = 0.0
            
        if curr_dist < prev_dist:
            reward += 0.5
        else:
            reward += -0.8
            
        if self.step_count > self.max_step:
            reward += 0 
            done = True
            
        if b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward += -100.0
            
        return reward, b_at_goal, done

    def _getRewardF(self, b_is_valid, prev_dist, curr_dist):
        """ Works well but does not optimize distance """
        done = False
        b_at_goal = 0
        reward = 0.0
        abs_dist = curr_dist
        if self._isComplete():
            reward += 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        else:
            # reward += 1.0/abs_dist**2 - 0.2
            reward = 0.0
            
        if curr_dist < prev_dist:
            reward += 1
            
        if self.step_count > self.max_step:
            reward += 0 
            done = True
            
        if b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward -= 0.0
            
        return reward, b_at_goal, done


    def _getRewardE(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        reward = 0.0
        abs_dist = curr_dist
        if self._isComplete():
            reward += 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        else:
            # reward += 1.0/abs_dist**2 - 0.2
            reward = 0.0
            
        # if curr_dist < prev_dist:
        #     reward += 0.1
            
        if self.step_count > self.max_step:
            reward += 0 
            done = True
            
        if b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward -= 1.0
            
        return reward, b_at_goal, done

    def _getRewardD(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        abs_dist = curr_dist
        if self._isComplete():
            reward = 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        elif self.step_count > self.max_step:
            reward = 0.5/abs_dist**2 - 0.5
            done = True
        elif b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward = 0.5/abs_dist**2 - 0.5 - 0.5 # self.violation_count**2
            # if self.violation_count > 5:
            #    done = True
        else:
            reward = 1.0/abs_dist**2 - 0.5
        return reward, b_at_goal, done

    def _getRewardC(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        abs_dist = curr_dist
        if self._isComplete():
            reward = 100.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        elif self.step_count > self.max_step:
            reward = 0.5/abs_dist**2 - 100.0
            done = True
        elif b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward = 0.5/abs_dist**2 - 1.0 - 0.5 # self.violation_count**2
            # if self.violation_count > 5:
            #    done = True
        else:
            reward = 1.0/abs_dist**2 - 1.0
        return reward, b_at_goal, done

    def _getRewardB(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        man_dist = self._getManhattenToGoal()
        if self._isComplete():
            reward = 1.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        elif self.step_count > self.max_step:
            reward = 1.0/man_dist
            done = True
        elif b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward = 1.0/man_dist
            done = False
        else:
            reward = 1.0/man_dist
        return reward, b_at_goal, done
    
    def _getRewardA(self, b_is_valid, prev_dist, curr_dist):
        done = False
        b_at_goal = 0
        if self._isComplete():
            reward = 10.0 #+ 0.5*(prev_dist-curr_dist)
            b_at_goal = 100
            done = True
        elif self.step_count > self.max_step:
            reward = -100
            done = True
        elif prev_dist > curr_dist:  # move toward the goal
            reward = 0.8*(prev_dist-curr_dist)
        elif b_is_valid == False: #action == Direction.ZZ: # penalize stopping
            reward = -100
            done = True
        elif prev_dist == curr_dist:
            reward = -0.8 * 6 # *(self.droplet[3]+self.droplet[2]-self.droplet[1]-self.droplet[0])
        elif curr_dist > prev_dist:  # move away the goal
            reward = -0.8*(curr_dist-prev_dist)
        else:
            raise Exception("Unknown reward condition - possible bug")
        return reward, b_at_goal, done

    # def _validAction(self, action):
    #     x_min,y_min,x_max,y_max = self.hazard
    #     x0,y0,x1,y1 = self.droplet
        
    #     if   action == Direction.NN and y1 >= y_max:
    #         tmpShift = [ 0,+1, 0,+1]
    #         tmpFront = self.m_degradation[x0:x1,y1]
    #         moveN = True
    #     elif action == Direction.SS and y0 > y_min:
    #         tmpShift = [ 0,-1, 0,-1]
    #         tmpFront = self.m_degradation[x0:x1,y0-1]
    #         moveS = True
    #     elif action == Direction.EE and x1 < x_max:
    #         tmpShift = [+1, 0,+1, 0]
    #         tmpFront = self.m_degradation[x1,y0:y1]
    #         moveE = True
    #     elif action == Direction.WW and x0 > x_min:
    #         tmpShift = [-1, 0,-1, 0]
    #         tmpFront = self.m_degradation[x0-1,y0:y1]
    #         moveW = True
    #     elif action == Direction.NE and y1 < y_max and x1 < x_max:
    #         tmpShift = [+1,+1,+1,+1]
    #         moveN, moveE = True, True
    #     elif action == Direction.NW and y1 < y_max and x0 > x_min:
    #         tmpShift = [-1,+1,-1,+1]
    #         moveN, moveW = True, True
    #     elif action == Direction.SE and y0 > y_min and x1 < x_max:
    #         tmpShift = [+1,-1,+1,-1]
    #         moveS, moveE = True, True
    #     elif action == Direction.SW and y0 > y_min and x0 > x_min:
    #         tmpShift = [-1,-1,-1,-1]
    #         moveS, moveW = True, True
    #     else:
    #         tmpShift = [ 0, 0, 0, 0]
    #         tmpFront = [1,]
                
    #     return b_is_valid

    def render(self, mode='human'):
        """Show environment
        """
        # screen_width = 600
        # screen_height = 400
        
        # if self.viewer is None:
        #     from gym.envs.classic_control import rendering
        #     self.viewer = rendering.Viewer(screen_width, screen_height)
        
        img = None
        if mode=='human':
            obs_full = self._getObs(full_res=True)
            plt.imshow(np.asarray(obs_full))
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)
            plt.savefig('log/_Render.png')
        elif mode=='human_frame':
            frame = self._getFrame()
            frame = np.transpose(frame,axes=[1,0,2])
            plt.imshow(frame)
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)
            plt.savefig('log/_Render.png',bbox_inches='tight')
        elif mode=='rgb_array':
            frame = self._getFrame()
            img = frame
        else:
            pass
        
        return img
        

    def close(self):
        """close render view
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        
        return


    # Private functions
    # [TODO][Cleaning] Remove _getCenter
    def _getCenter(self, dr):
        return ( (dr[0]+dr[2])/2.0 , (dr[1]+dr[3])/2.0 )
    
    
    def _getDistanceToGoal(self):
        # [NOTE] Use L2 norm instead of this mess
        # dist = np.mean(np.abs(self.goal - self.droplet))
        # Ecludian distance
        # dist = np.linalg.norm(self.goal[0:2] - self.droplet[0:2])
        # Manhatten distance
        dist = np.abs(self.goal[0]-self.droplet[0]) + \
            np.abs(self.goal[1]-self.droplet[1])
        return dist
    
    def _getManhattenToGoal(self):
        dist = np.abs(self.goal[0]-self.droplet[0]) + \
            np.abs(self.goal[1]-self.droplet[1])
        return dist
    
    def _getStratifiedSample(self):
        # Fill buffers
        if len(self.x_set_buff)==0:
            self.x_set_buff = copy.copy(self.x_set)
            random.shuffle(self.x_set_buff)
        if len(self.y_set_buff)==0:
            self.y_set_buff = copy.copy(self.y_set)
            random.shuffle(self.y_set_buff)
        if len(self.dr_size_set_buff)==0:
            self.dr_size_set_buff = copy.copy(self.dr_size_set)
            random.shuffle(self.dr_size_set_buff)
            
        # Randomly select droplet size first
        dr_width, dr_height = self.dr_size_set_buff.pop(-1)
        x0max, y0max = self.width-dr_width, self.height-dr_height
        xg = self.x_set_buff.pop(-1)
        xs = self.x_set_buff.pop(-1)
        yg = self.y_set_buff.pop(-1)
        ys = self.y_set_buff.pop(-1)
        
        # Randomly generate goal state
        xg0, yg0 = min(max(0,xg-dr_width//2),x0max), min(max(0,yg-dr_height//2),y0max)
        xg1, yg1 = xg0 + dr_width, yg0 + dr_height
        self.goal = np.array((xg0,yg0,xg1,yg1))
        
        # Randomly generate droplet state
        xs0, ys0 = min(max(0,xs-dr_width//2),x0max), min(max(0,ys-dr_height//2),y0max)
        xs1, ys1 = xs0+dr_width, ys0+dr_height # includes mins and excludes maxs
        self.droplet = np.array((xs0,ys0,xs1,ys1))
        return
    
    def _getUniformSample(self):
        # Randomly select droplet size first
        dr_width, dr_height = random.choice(self.droplet_sizes)
        x0max, y0max = self.width-dr_width, self.height-dr_height
        
        # Randomly generate goal state
        xg0, yg0 = random.randint(0,x0max), random.randint(0,y0max)
        xg1, yg1 = xg0 + dr_width, yg0 + dr_height
        self.goal = np.array((xg0,yg0,xg1,yg1))
        
        # Randomly generate droplet state
        while True:
            xs0, ys0 = random.randint(0, x0max), random.randint(0, y0max)
            if xs0 != xg0 or ys0 != yg0: break # prevents starting at goal
        xs1, ys1 = xs0+dr_width, ys0+dr_height # includes mins and excludes maxs
        self.droplet = np.array((xs0,ys0,xs1,ys1))
        return
        
        
    def _resetInitialState(self):
        """Returns droplet start and goal locations. Gets called upon
        initialization and reset.
        """
        # [DONE] Implement random dr_0 and dr_g generator based on size
        
        self.fn_get_sample()
        
        # Compute hazard bounds
        self.hazard[:] = (
            max(min(self.goal[0],self.droplet[0])-3, 0),
            max(min(self.goal[1],self.droplet[1])-3, 0),
            min(max(self.goal[2],self.droplet[2])+3, self.width),
            min(max(self.goal[3],self.droplet[3])+3, self.height)
        )
        
        # xmin,ymin,xmax,ymax = min(xg0,xs0),min(yg0,ys0),max(xg1,xs1),max(yg1,ys1)
        # self.hazard = np.clip([xmin,ymin,xmax,ymax],)
        # self.hazard[:] = (
        #     max(min(self.goal[0],self.droplet[0])-3,0),
        #     max(min(self.goal[1],self.droplet[1])-3,0),
        #     min(max(self.goal[2],self.droplet[2])+3,self.width),
        #     min(max(self.goal[3],self.droplet[3])+3,self.height)
        # )
        # xs0 = random.randint(self.width-dr_width-2, self.width-dr_width)
        # ys0 = random.randint(self.height-dr_height-2, self.height-dr_height)
        # xg0, yg0 = 0, 0
        # xg0 = random.randint(1,self.width-dr_width)
        # yg0 = random.randint(1,np.min((xg0,self.height-dr_height)))
        # [NOTE][2021-07-25] This was replaced by full range
        # xs0 = random.choice(self.xs0range)
        # ys0 = random.choice(self.ys0range)
        # b_tmp_at_goal = True
        # while b_tmp_at_goal: # prevents starting at goal
        #     xs0 = random.randint(0,self.width-dr_width)
        #     ys0 = random.randint(0,self.height-dr_height)
        #     b_tmp_at_goal = (xs0==xg0) and (ys0==yg0)
        
        
        
        # Set the max number of steps allowed
        # self.max_step = 2*(self.hazard[2]-self.hazard[0]+self.hazard[3]-self.hazard[1])
        self.max_step = 1*(self.width+self.height)
        
        self.collision.fill(False)
        self.action = Direction(0)
        
        # [TODO] Remove agt_sta and disable old router
        # self.agt_sta = copy.deepcopy(self.droplet)
        return
    
    
    def _resetInitialHealth(self):
        """Resets initial health parameters
        """
        self.m_taus.fill(0.7)
        self.m_C1s.fill(0)
        self.m_C2s.fill(200)
        if self.deg_mode=='normal':
            pass
        elif self.deg_mode=='random':
            items = np.random.choice(
                self.m_C1s.size, int(self.m_C1s.size * self.deg_perc/self.deg_size),
                replace=False)
            for idx in items:
                r,c = idx // self.m_C1s.shape[1], idx % self.m_C1s.shape[1]
                self.m_C1s[r:r+self.deg_size, c:c+self.deg_size].fill(np.Inf)
        else:
            raise Exception("Unknown degradation mode: %s" % self.deg_mode)
        return
        
    
    def _resetActuationMatrix(self, is_random=False):
        """Resets actuation matrix
        """
        self.m_pattern.fill(0)
        if is_random:
            self.m_actcount = random.randint(0,1000)
            self.m_actcount = np.random.randint(
                0,2000,(self.width, self.height)
            )
        else:
            self.m_actcount = np.zeros((self.width, self.height))
        return
    
    
    def _updatePattern(self, action):
        """Update actuation pattern
        """
        x_min,y_min,x_max,y_max = self.hazard
        x0,y0,x1,y1 = self.droplet
        b_is_valid = True
        # Flags that indicate whether to check movement in a given direction:
        moveN, moveS, moveE, moveW = False, False, False, False
        # Find the shift in position
        if   action == Direction.NN and y1 < y_max:
            tmpShift = [ 0,+1, 0,+1]
            tmpFront = self.m_degradation[x0:x1,y1]
            moveN = True
        elif action == Direction.SS and y0 > y_min:
            tmpShift = [ 0,-1, 0,-1]
            tmpFront = self.m_degradation[x0:x1,y0-1]
            moveS = True
        elif action == Direction.EE and x1 < x_max:
            tmpShift = [+1, 0,+1, 0]
            tmpFront = self.m_degradation[x1,y0:y1]
            moveE = True
        elif action == Direction.WW and x0 > x_min:
            tmpShift = [-1, 0,-1, 0]
            tmpFront = self.m_degradation[x0-1,y0:y1]
            moveW = True
        elif action == Direction.NE and y1 < y_max and x1 < x_max:
            tmpShift = [+1,+1,+1,+1]
            moveN, moveE = True, True
        elif action == Direction.NW and y1 < y_max and x0 > x_min:
            tmpShift = [-1,+1,-1,+1]
            moveN, moveW = True, True
        elif action == Direction.SE and y0 > y_min and x1 < x_max:
            tmpShift = [+1,-1,+1,-1]
            moveS, moveE = True, True
        elif action == Direction.SW and y0 > y_min and x0 > x_min:
            tmpShift = [-1,-1,-1,-1]
            moveS, moveW = True, True
        # elif action == Direction.ZZ:
        #     tmpShift = [ 0, 0, 0, 0]
        #     tmpFront = [1,]
        #     b_is_valid = False
        else:
            self.collision[0] = (x0==x_min) and (action in {Direction.SW,Direction.WW,Direction.NW})
            self.collision[1] = (y0==y_min) and (action in {Direction.SE,Direction.SS,Direction.SW})
            self.collision[2] = (x1==x_max) and (action in {Direction.NE,Direction.EE,Direction.SE})
            self.collision[3] = (y1==y_max) and (action in {Direction.NW,Direction.NN,Direction.NE})
            tmpShift = [ 0, 0, 0, 0]
            tmpFront = [1,]
            b_is_valid = False
            
        # Compute new pattern
        if self.b_parm_step:
            # First, compute droplet radius
            self.tmpRadius = np.floor_divide(self.droplet[[2,3]]-self.droplet[[0,1]],2)
            # Shift as much as the radius
            tmpShift = tmpShift * self.tmpRadius[[0,1,0,1]]
            tmpDr = self.droplet + tmpShift
            # Prevent target overshooting in y-axis
            if (y0 < self.goal[1] < tmpDr[1]) or (y0 > self.goal[1] > tmpDr[1]):
                tmpDr[[1,3]] = self.goal[[1,3]]
            # Prevent target overshooting in x-axis
            if (x0 < self.goal[0] < tmpDr[0]) or (x0 > self.goal[0] > tmpDr[0]):
                tmpDr[[0,2]] = self.goal[[0,2]]
        else:
            tmpDr = self.droplet + tmpShift # New droplet location
        # self.m_prev_pattern = np.copy(self.m_pattern)
        # Correct tmpDr if needed
        if   tmpDr[0] < x_min: tmpDr[[0,2]] += x_min - tmpDr[[0,0]] #[BUG#0001]
        elif tmpDr[2] > x_max: tmpDr[[0,2]] -= tmpDr[[2,2]] - x_max
        if   tmpDr[1] < y_min: tmpDr[[1,3]] += y_min - tmpDr[[1,1]] #[BUG#0001]
        elif tmpDr[3] > y_max: tmpDr[[1,3]] -= tmpDr[[3,3]] - y_max
        # Reset pattern, np.array.fill is faster than [:,:] = 0
        self.m_pattern.fill(0)
        # Note: if tmpDr range is beyond m_pattern, the outskirts are ignored
        self.m_pattern[tmpDr[0]:tmpDr[2],tmpDr[1]:tmpDr[3]] = 1 # New pattern
        
        # Apply probabilistic movements
        dr = self.droplet
        while (moveN or moveS or moveE or moveW):
            # x0,y0,x1,y1 = self.droplet
            probN, probS, probE, probW = 0, 0, 0, 0
            if moveN:
                if dr[3] < y_max:
                    frontSetN = self.m_pattern[max(dr[0]-1,0):dr[2]+1,dr[3]]
                    degraSetN = self.m_degradation[max(dr[0]-1,0):dr[2]+1,dr[3]]
                    frontN = frontSetN.sum() #[BUG#0002]
                    # frontN = self.m_pattern[dr[0]-1:dr[2]+1,dr[3]].sum()
                    if frontN > 0: probN = (np.dot(frontSetN, degraSetN) / frontN)
                    else: probN = 0
                moveN = ( random.random() <= probN )
            elif moveS:
                if dr[1] > y_min:
                    frontSetS = self.m_pattern[max(dr[0]-1,0):dr[2]+1,dr[1]-1]
                    degraSetS = self.m_degradation[max(dr[0]-1,0):dr[2]+1,dr[1]-1]
                    frontS = frontSetS.sum()
                    if frontS > 0: probS = (np.dot(frontSetS, degraSetS) / frontS)
                    else: probS = 0
                moveS = ( random.random() <= probS )
            if moveE:
                if dr[2] < x_max:
                    frontSetE = self.m_pattern[dr[2],max(dr[1]-1,0):dr[3]+1]
                    degraSetE = self.m_degradation[dr[2],max(dr[1]-1,0):dr[3]+1]
                    frontE = frontSetE.sum()
                    if frontE > 0: probE = (np.dot(frontSetE, degraSetE) / frontE)
                    else: probE = 0
                moveE = ( random.random() <= probE )
            elif moveW:
                if dr[0] > x_min:
                    frontSetW = self.m_pattern[dr[0]-1,max(dr[1]-1,0):dr[3]+1]
                    degraSetW = self.m_degradation[dr[0]-1,max(dr[1]-1,0):dr[3]+1]
                    frontW = frontSetW.sum()
                    if frontW > 0: probW = (np.dot(frontSetW, degraSetW) / frontW)
                    else: probW = 0
                moveW = ( random.random() <= probW )
            # Move droplet
            if moveN: dr += [ 0,+1, 0,+1]
            if moveS: dr += [ 0,-1, 0,-1]
            if moveE: dr += [+1, 0,+1, 0]
            if moveW: dr += [-1, 0,-1, 0]
        
        self.prev_m_pattern[:] = self.m_pattern[:]
        return b_is_valid
        
        
    def _updateHealth(self):
        """Updates the degradation and health matrices
        """
        # Abort if no degradation required
        if not self.b_degrade:
            return
        # Update degradation matrix based on no. actuations and parameters
        self.m_degradation = self.m_taus ** (
            (self.m_actcount + self.m_C1s) / self.m_C2s )
        # Update health matrix from degradation matrix
        self.m_health = (
            (np.ceil(self.m_degradation*(2**self.n_bits)-0.5))/(2**self.n_bits) )
        return
    
    
    def _getObs(self, full_res=False):
        # obs = np.zeros(shape=(self.height, self.width, self.n_layers))
        # obs = self._addModulesInObs(obs)
        obs = np.zeros_like(self.default_observation)
        # Goal layer
        x0,y0,x1,y1 = self.goal
        obs[x0:x1,y0:y1,self.goal_layer_id] = 1
        # Droplet layer
        x0,y0,x1,y1 = self.droplet
        obs[x0:x1,y0:y1,self.droplet_layer_id] = 1
        # Collision
        if self.collision[0]:
            obs[x0,y0:y1,self.droplet_layer_id] = 0.5
        if self.collision[1]: 
            obs[x0:x1,y0,self.droplet_layer_id] = 0.5
        if self.collision[2]:
            obs[x1-1,y0:y1,self.droplet_layer_id] = 0.5
        if self.collision[3]:
            obs[x0:x1,y1-1,self.droplet_layer_id] = 0.5
        # Health layer, this leaves health values outside hazard bounds as 0s
        x0,y0,x1,y1 = self.hazard
        obs[x0:x1,y0:y1,self.health_layer_id] = self.m_health[x0:x1,y0:y1]
        # Resize obs if needed
        if self.b_unify_obs and not full_res:
            obs = cv2.resize(obs, self.obs_size, interpolation=cv2.INTER_AREA)
        return obs
    
    
    def _getFrame(self):
        frame = np.zeros(shape=(self.width, self.height, 3), dtype=np.float)
        # Add degradation (gray)
        frame[:,:,0] = self.m_degradation
        frame[:,:,1] = self.m_degradation
        frame[:,:,2] = self.m_degradation
        x0,y0,x1,y1 = self.hazard
        mask = np.ones_like(frame)
        mask[:,:,[1,2]] *=0.5
        mask[x0:x1,y0:y1,:].fill(1)
        frame *= mask
        # Add Goal (green)
        x0,y0,x1,y1 = self.goal
        frame[x0:x1,y0:y1,0] = 0
        frame[x0:x1,y0:y1,1] = 1
        frame[x0:x1,y0:y1,2] = 0
        # Add Droplet (B)
        x0,y0,x1,y1 = self.droplet
        frame[x0:x1,y0:y1,0] = 0
        frame[x0:x1,y0:y1,1] = 0
        frame[x0:x1,y0:y1,2] = 1
        # Correct orientation
        frame = np.flip(frame,1)
        return frame
    
    
    def _isComplete(self):
        if np.array_equal(self.droplet, self.goal):
            return True
        else:
            return False

