import copy
import random
import numpy as np
from enum import IntEnum
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
    

class Direction(IntEnum):
    ZZ = 0  # Sleep/Stop
    NN = 1  
    NE = 2  
    EE = 3  
    SE = 4  
    SS = 5  
    SW = 6  
    WW = 7  
    NW = 8  


class MEDAEnv(gym.Env):
    """ MEDA biochip environment, following gym interface """
    metadata = {'render.modes': ['human', 'human_frame', 'rgb_array'],
                'video.frames_per_second': 2}

    def __init__(self, width=0, height=0, droplet_sizes=[[4,4],], n_bits=2,
                 b_degrade=True, b_use_dict=False,
                 b_unify_obs=True, b_parm_step=True, obs_size=(30, 30),
                 deg_mode='normal', deg_perc=0.1, deg_size=1,
                 b_play_mode=False, delay_counter = 10,
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
        self.xs0range = [3,]
        self.ys0range = [1,3,]
        self.act_count = np.zeros(9)
        
        # State vars
        self.droplet = np.zeros(4, dtype=np.int)
        self.goal = np.zeros(4, dtype=np.int)
        self.hazard = np.zeros(4, dtype=np.int)
        self.m_taus = np.ones((width, height)) * 0.8
        self.m_C1s = np.ones((width, height)) * 0
        self.m_C2s = np.ones((width, height)) * 200
        self.m_degradation = np.ones((width, height))
        self.m_health = np.ones((width, height))
        self.m_actuations_count = np.zeros((width, height))
        self.m_pattern = np.zeros((width, height), dtype=np.uint8)
        self.step_count = 0
        self.max_step = 0

        # Gym environment: rewards and action space
        self.reward_range = (-100.0, 1.0)
        self.action_space = spaces.Discrete(len(self.actions))
        if b_use_dict:
            self.observation_space = spaces.Dict({
                'health': spaces.Box(
                    low=0, high=2**n_bits-1,
                    shape=(width, height, 1), dtype=np.uint8),
                'sensors': spaces.Box(
                    low=0, high=1,
                    shape=(width, height), dtype=np.uint8)
            })
            self.keys = list(self.observation_space.spaces.keys())
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
                    shape=(30, 30, self.n_layers), dtype=np.float)
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
        
        # Reset initial state
        self._resetActuationMatrix()
        self._resetInitialState()
        self._resetInitialHealth()
        self._updateHealth()
        
        return

    
    def get_keys_to_action(self):
        return self.keys_to_action


    # Gym Interfaces
    def reset(self):
        """Reset environment state 

        Returns:
            obs: Observation
        """
        self.step_count = 0
        self._resetActuationMatrix(is_random=False)
        self._resetInitialState()
        self._resetInitialHealth()
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
        self._updatePattern(action)
        curr_dist = self._getDistanceToGoal()
        self.m_actuations_count += self.m_pattern
        self._updateHealth()
        obs = self._getObs()
        done = False
        b_at_goal = 0
        if self._isComplete():
            reward = 1.0
            b_at_goal = 100
            done = True
        elif self.step_count > self.max_step:
            reward = -100
            done = True
        elif prev_dist > curr_dist:  # move toward the goal
            reward = 0.5*(prev_dist-curr_dist)
        elif action == Direction.ZZ: # penalize stopping
            reward = -100
            done = True
        elif prev_dist == curr_dist:
            reward = -0.3
        else:  # move away the goal
            reward = -0.8*(curr_dist-prev_dist)
        
            
        return obs, reward, done, {"b_at_goal":b_at_goal, "num_cycles":self.step_count}


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
            plt.savefig('log/Render.png')
        elif mode=='human_frame':
            frame = self._getFrame()
            plt.imshow(np.asarray(frame))
            plt.axis('off')
            plt.draw()
            plt.pause(0.001)
            plt.savefig('log/Render.png')
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
        dist = np.mean(np.abs(self.goal - self.droplet))
        return dist
        
        
    def _resetInitialState(self):
        """Returns droplet start and goal locations. Gets called upon
        initialization and reset.
        """
        # [DONE] Implement random dr_0 and dr_g generator based on size
        # Randomly select droplet size first
        dr_width, dr_height = random.choice(self.droplet_sizes)
        xs0 = random.choice(self.xs0range)
        ys0 = random.choice(self.ys0range)
        # Droplet range includes mins and excludes maxs
        xs1 = xs0 + dr_width
        ys1 = ys0 + dr_height
        self.droplet = np.array((xs0,ys0,xs1,ys1))
        
        # Randomly generate goal state
        xg0 = random.randint(1,self.width-dr_width)
        yg0 = random.randint(1,np.min((xg0,self.height-dr_height)))
        xg1 = xg0 + dr_width
        yg1 = yg0 + dr_height
        self.goal[:] = (xg0,yg0,xg1,yg1)
        
        # Compute hazard bounds
        self.hazard[:] = (
            max(min(self.goal[0],self.droplet[0])-3,0),
            max(min(self.goal[1],self.droplet[1])-3,0),
            min(max(self.goal[2],self.droplet[2])+3,self.width),
            min(max(self.goal[3],self.droplet[3])+3,self.height)
        )
        
        # Set the max number of steps allowed
        self.max_step = 2*(self.hazard[2]-self.hazard[0]+self.hazard[3]-self.hazard[1])
        
        # [TODO] Remove agt_sta and disable old router
        # self.agt_sta = copy.deepcopy(self.droplet)
        return
    
    
    def _resetInitialHealth(self):
        """Resets initial health parameters
        """
        self.m_taus.fill(0.8)
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
            self.m_actuations_count = random.randint(0,1000)
            self.m_actuations_count = np.random.randint(
                0,2000,(self.width, self.height)
            )
        else:
            self.m_actuations_count = np.zeros((self.width, self.height))
        return
    
    
    def _updatePattern(self, action):
        """Update actuation pattern
        """
        x_min,y_min,x_max,y_max = self.hazard
        x0,y0,x1,y1 = self.droplet
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
        else:
            tmpShift = [ 0, 0, 0, 0]
            tmpFront = [1,]
            
        # Compute new pattern
        if self.b_parm_step:
            # First, compute droplet radius
            tmpRadius = np.floor_divide(self.droplet[[2,3]]-self.droplet[[0,1]],2)
            # Shift as much as the radius
            tmpShift = tmpShift * tmpRadius[[0,1,0,1]]
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
        if   tmpDr[0] < x_min: tmpDr[[0,2]] -= x_min - tmpDr[[0,0]]
        elif tmpDr[2] > x_max: tmpDr[[0,2]] -= tmpDr[[2,2]] - x_max
        if   tmpDr[1] < y_min: tmpDr[[1,3]] -= y_min - tmpDr[[1,1]]
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
                    frontN = self.m_pattern[dr[0]-1:dr[2]+1,dr[3]].sum()
                    if frontN > 0:
                        probN = (
                            np.dot(self.m_pattern[dr[0]-1:dr[2]+1,dr[3]],
                                self.m_degradation[dr[0]-1:dr[2]+1,dr[3]]) /
                            frontN )
                    else:
                        probN = 0
                moveN = ( random.random() <= probN )
            elif moveS:
                if dr[1] > y_min:
                    frontS = self.m_pattern[dr[0]-1:dr[2]+1,dr[1]-1].sum()
                    if frontS > 0:
                        probS = (
                            np.dot(self.m_pattern[dr[0]-1:dr[2]+1,dr[1]-1],
                                self.m_degradation[dr[0]-1:dr[2]+1,dr[1]-1]) /
                            frontS )
                    else:
                        probS = 0
                moveS = ( random.random() <= probS )
            if moveE:
                if dr[2] < x_max:
                    frontE = self.m_pattern[dr[2],dr[1]-1:dr[3]+1].sum()
                    if frontE > 0:
                        probE = (
                            np.dot(self.m_pattern[dr[2],dr[1]-1:dr[3]+1],
                                self.m_degradation[dr[2],dr[1]-1:dr[3]+1]) /
                            frontE )
                    else:
                        probE = 0
                moveE = ( random.random() <= probE )
            elif moveW:
                if dr[0] > x_min:
                    frontW = self.m_pattern[dr[0]-1,dr[1]-1:dr[3]+1].sum()
                    if frontW > 0:
                        probW = (
                            np.dot(self.m_pattern[dr[0]-1,dr[1]-1:dr[3]+1],
                                self.m_degradation[dr[0]-1,dr[1]-1:dr[3]+1]) /
                            frontW )
                    else:
                        probW = 0
                moveW = ( random.random() <= probW )
            if moveN: dr += [ 0,+1, 0,+1]
            if moveS: dr += [ 0,-1, 0,-1]
            if moveE: dr += [+1, 0,+1, 0]
            if moveW: dr += [-1, 0,-1, 0]

        # Update position
        # if self.b_degrade:
        #     prob = np.mean(tmpFront)
        # else:
        #     prob = 1
        # # Draw a sample from the bernolli distribution Bern(prob)
        # if random.random() <= prob:
        #     # Random sample is True
        #     self.droplet += tmpShift
        # else:
        #     # Random sample is False
        #     pass
        
        return
        
        
    def _updateHealth(self):
        """Updates the degradation and health matrices
        """
        # Abort if no degradation required
        if not self.b_degrade:
            return
        # Update degradation matrix based on no. actuations and parameters
        self.m_degradation = self.m_taus ** (
            (self.m_actuations_count + self.m_C1s) / self.m_C2s )
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

