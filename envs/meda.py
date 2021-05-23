import copy
import math
import queue
import random
import numpy as np
from PIL import Image
from enum import IntEnum

import gym
from gym import error, spaces, utils
# from gym.wrappers import ResizeObservation
import cv2
from gym.utils import seeding


class DirectionSimple(IntEnum):
    NN = 0  # North
    EE = 1  # East
    SS = 2  # South
    WW = 3  # West
    

class Direction(IntEnum):
    NN = 0  # North
    EE = 1  # East
    SS = 2  # South
    WW = 3  # West
    NE = 4  # N. East
    NW = 5  # N. West
    SE = 6  # S. East
    SW = 7  # S. West



class Module:
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Module() inputs are illegal')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


    def isPointInside(self, point):
        ''' point is in the form of (y, x) '''
        if point[0] >= self.y_min and point[0] <= self.y_max and \
                point[1] >= self.x_min and point[1] <= self.x_max:
            return True
        else:
            return False


    def isModuleOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and \
                self._isLinesOverlap(self.y_min, self.y_max, m.y_min, m.y_max):
            return True
        else:
            return False


    def _isLinesOverlap(self, xa_1, xa_2, xb_1, xb_2):
        if xa_1 > xb_2:
            return False
        elif xb_1 > xa_2:
            return False
        else:
            return True



class MEDAEnv(gym.Env):
    """ MEDA biochip environment, following gym interface """
    metadata = {'render.modes': ['human']}

    def __init__(self, width=0, height=0, droplet_sizes=[[4,4],], n_bits=2,
                 b_degrade=True, per_degrade=0.1, b_use_dict=False,
                 b_unify_obs=True, b_parm_step=True,
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
        width, height = kwargs['size']
        self.reset_counter = 0
        # Instance variables
        self.height = height
        self.width = width
        assert height > 4 and width > 4
        self.n_bits = n_bits
        self.b_unify_obs = b_unify_obs
        self.b_parm_step = b_parm_step
        self.actions = Direction
        # Degradation parameters
        self.b_degrade = b_degrade
        self.m_taus = np.ones((width, height)) * 0.8
        self.m_C1s = np.ones((width, height)) * 0
        self.m_C2s = np.ones((width, height)) * 200
        # Degradation matrix
        self.m_degradation = np.ones((width, height))
        # Health matrix
        self.m_health = np.ones((width, height))
        # Actuations count matrix
        self.m_actuations_count = np.zeros((width, height))
        # Control pattern
        self.m_pattern = np.zeros((width, height), dtype=np.uint8)
        # self.m_prev_pattern = np.zeros((width, height))
        # Number of steps 
        self.step_count = 0
        # Maximum number of steps
        self.max_step = 2 * (height + width)
        # List of droplet sizes
        self.droplet_sizes = droplet_sizes
        # Droplet range
        self.xs0range = [3,]
        self.ys0range = [1,3,]

        # Gym environment: action space
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
            # Layers: (
                # 0: obstacles,
                # 1: goal,
                # 2: sensors,
                # 3: health )
            self.n_layers = 3 + 1
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
                
        self.reward_range = (-1.0, 1.0)
        
        # Reset actuations matrix
        self._resetActuationMatrix()
        # Get new start and goal locations and reset initial state
        self._resetInitialState()
        # Update health
        self._updateHealth()
        # Update distance to goal
        self.m_distance = self._getDistanceToGoal()
        
        return


    # Gym Interfaces
    def reset(self):
        """Reset environment state 

        Returns:
            obs: Observation
        """
        self.reset_counter += 1
        # Reset steps counter
        self.step_count = 0
        # Reset actuations matrix
        self._resetActuationMatrix(is_random=False)
        # Get new start and goal locations and reset initial state
        self._resetInitialState()
        # Reset actuation pattern
        self.m_pattern[:,:] = 0
        # self.m_prev_pattern[:,:] = 0
        # Update health
        self._updateHealth()
        self.m_distance = self._getDistanceToGoal()
        obs = self._getObs()
        # print("Reset #%5d" % self.reset_counter)
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
        self.step_count += 1
        # Update actuation pattern and compute new position
        prev_dist = self._getDistanceToGoal()
        self._updatePattern(action)
        self.agt_sta = copy.deepcopy(self.droplet)
        curr_dist = self._getDistanceToGoal()
        # Update the global number of actuations
        self.m_actuations_count += self.m_pattern
        # Update biochop health
        self._updateHealth()
        # Update observation
        obs = self._getObs()
        # Compute rewards
        done = False
        b_at_goal = 0
        if self._isComplete():
            reward = 1.0
            b_at_goal = 1
            done = True
        elif self.step_count > self.max_step:
            reward = -0.8
            done = True
        elif prev_dist > curr_dist:  # move toward the goal
            reward = 0.5
        elif prev_dist == curr_dist:
            reward = -0.3
        else:  # move away the goal
            reward = -0.8
        return obs, reward, done, {"b_at_goal":b_at_goal, "num_cycles":self.step_count}


    def render(self, mode='human'):
        """Show environment
        """
        # goal:2, pos:1, blocks:-1, degrade: -2
        if mode == 'human':
            img = np.zeros(shape= \
                               (self.height, self.width))
            img[self.goal[0]][self.goal[1]] = 2
            img[self.droplet[0]][self.droplet[1]] = 1
            for m in self.modules:
                for y in range(m.y_min, m.y_max + 1):
                    for x in range(
                            m.x_min, m.x_max + 1):
                        img[y][x] = -1
            if self.b_degrade:
                img[self.m_health < 0.5] = -2
            return img
        elif mode == 'rgb_array':
            img = self._getObs().astype(np.uint8)
            for y in range(self.height):
                for x in range(self.width):
                    if np.array_equal(img[y][x], [1, 0, 0]):  # red
                        img[y][x] = [255, 0, 0]
                    elif np.array_equal(img[y][x], [0, 1, 0]):  # gre
                        img[y][x] = [0, 255, 0]
                    elif np.array_equal(img[y][x], [0, 0, 1]):  # blu
                        img[y][x] = [0, 0, 255]
                    elif self.b_degrade and \
                            self.m_health[y][x] < 0.5:  # ppl
                        img[y][x] = [255, 102, 255]
                    elif self.b_degrade and \
                            self.m_health[y][x] < 0.7:  # ppl
                        img[y][x] = [255, 153, 255]
                    else:  # grey
                        img[y][x] = [192, 192, 192]
            return img
        else:
            raise RuntimeError(
                'Unknown mode in render')


    def close(self):
        """ close render view """
        pass


    # Private functions
    def _getCenter(self, dr):
        return ( (dr[0]+dr[2])/2.0 , (dr[1]+dr[3])/2.0 )
    
    
    def _getDistanceToGoal(self):
        dist = np.mean(np.abs(self.goal - self.droplet))
        return dist
        
        
    def _resetInitialState(self):
        """Returns droplet start and goal locations. Gets called upon
        initialization and reset.
        """
        # [TODO] Implement random dr_0 and dr_g generator based on size
        # Select droplet size first
        dr_width = self.droplet_sizes[0][0]
        dr_height = self.droplet_sizes[0][1]
        # Randomly generate initial state
        # self.droplet = np.array((3,2,6,5))
        xs0 = random.choice(self.xs0range)
        ys0 = random.choice(self.ys0range)
        # [TODO] Check whether droplet range is closed or half open
        xs1 = xs0 + dr_width
        ys1 = ys0 + dr_height
        self.droplet = np.array((xs0,ys0,xs1,ys1))
        
        # Randomly generate goal state
        # self.goal = np.array((self.width-4-3,self.height-3-3,self.width-4,self.height-3))
        xg0 = random.randint(1,self.width-dr_width)
        yg0 = random.randint(1,np.min((xg0,self.height-dr_height)))
        xg1 = xg0 + dr_width
        yg1 = yg0 + dr_height
        self.goal = np.array((xg0,yg0,xg1,yg1))
        
        self.agt_sta = copy.deepcopy(self.droplet)
        return
    
    
    def _resetActuationMatrix(self, is_random=False):
        """Resets actuation matrix
        """
        if is_random:
            self.m_actuations_count = np.random.randint(
                0,2000,(self.width, self.height)
            )
        else:
            self.m_actuations_count = np.zeros((self.width, self.height))
        return
    
    
    def _updatePattern(self, action):
        """Update actuation pattern
        """
        x0,y0,x1,y1 = self.droplet
        # Flags that indicate whether to check movement in a given direction:
        moveN, moveS, moveE, moveW = False, False, False, False
        # Find the shift in position
        if   action == Direction.NN and y1 < self.height:
            tmpShift = [ 0,+1, 0,+1]
            tmpFront = self.m_degradation[x0:x1,y1]
            moveN = True
        elif action == Direction.SS and y0 > 0:
            tmpShift = [ 0,-1, 0,-1]
            tmpFront = self.m_degradation[x0:x1,y0-1]
            moveS = True
        elif action == Direction.EE and x1 < self.width:
            tmpShift = [+1, 0,+1, 0]
            tmpFront = self.m_degradation[x1,y0:y1]
            moveE = True
        elif action == Direction.WW and x0 > 0:
            tmpShift = [-1, 0,-1, 0]
            tmpFront = self.m_degradation[x0-1,y0:y1]
            moveW = True
        elif action == Direction.NE and y1 < self.height and x1 < self.width:
            tmpShift = [+1,+1,+1,+1]
            moveN, moveE = True, True
        elif action == Direction.NW and y1 < self.height and x0 > 0:
            tmpShift = [-1,+1,-1,+1]
            moveN, moveW = True, True
        elif action == Direction.SE and y0 > 0 and x1 < self.width:
            tmpShift = [+1,-1,+1,-1]
            moveS, moveE = True, True
        elif action == Direction.SW and y0 > 0 and x0 > 0:
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
        if   tmpDr[0] < 0:           tmpDr[[0,2]] -= tmpDr[[0,0]]
        elif tmpDr[2] > self.width:  tmpDr[[0,2]] -= tmpDr[[2,2]] - self.width
        if   tmpDr[1] < 0:           tmpDr[[1,3]] -= tmpDr[[1,1]]
        elif tmpDr[3] > self.height: tmpDr[[1,3]] -= tmpDr[[3,3]] - self.height
        # Reset pattern
        self.m_pattern[:,:] = 0
        # Note: if tmpDr range is beyond m_pattern, the outskirts are ignored
        self.m_pattern[tmpDr[0]:tmpDr[2],tmpDr[1]:tmpDr[3]] = 1 # New pattern
        
        # Apply probabilistic movements
        dr = self.droplet
        while (moveN or moveS or moveE or moveW):
            # x0,y0,x1,y1 = self.droplet
            probN, probS, probE, probW = 0, 0, 0, 0
            if moveN:
                if dr[3] < self.height:
                    probN = (
                        np.dot(self.m_pattern[dr[0]-1:dr[2]+1,dr[3]],
                            self.m_degradation[dr[0]-1:dr[2]+1,dr[3]]) /
                        self.m_pattern[dr[0]-1:dr[2]+1,dr[3]].sum() )
                moveN = ( random.random() <= probN )
            elif moveS:
                if dr[1] > 0:
                    probS = (
                        np.dot(self.m_pattern[dr[0]-1:dr[2]+1,dr[1]-1],
                            self.m_degradation[dr[0]-1:dr[2]+1,dr[1]-1]) /
                        self.m_pattern[dr[0]-1:dr[2]+1,dr[1]-1].sum() )
                moveS = ( random.random() <= probS )
            if moveE:
                if dr[2] < self.width:
                    probE = (
                        np.dot(self.m_pattern[dr[2],dr[1]-1:dr[3]+1],
                            self.m_degradation[dr[2],dr[1]-1:dr[3]+1]) /
                        self.m_pattern[dr[2],dr[1]-1:dr[3]+1].sum() )
                moveE = ( random.random() <= probE )
            elif moveW:
                if dr[0] > 0:
                    probW = (
                        np.dot(self.m_pattern[dr[0]-1,dr[1]-1:dr[3]+1],
                            self.m_degradation[dr[0]-1,dr[1]-1:dr[3]+1]) /
                        self.m_pattern[dr[0]-1,dr[1]-1:dr[3]+1].sum() )
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
    
    
    def _getObs(self):
        """
        0: Obstacles,
        1: Goal,
        2: Droplet,
        3: Health
        """
        # obs = np.zeros(shape=(self.height, self.width, self.n_layers))
        # obs = self._addModulesInObs(obs)
        obs = np.zeros_like(self.default_observation)
        # Goal layer
        x0,y0,x1,y1 = self.goal
        obs[y0:y1+1,x0:x1+1,1] = 1
        # Droplet layer
        x0,y0,x1,y1 = self.droplet
        obs[y0:y1+1,x0:x1+1,2] = 1
        # Health layer
        obs[:,:,3] = self.m_health
        if self.b_unify_obs:
            obs = cv2.resize(
                obs, (30, 30),
                interpolation=cv2.INTER_AREA
            )
        return obs
    
    
    def _getUnifiedObs(self):
        """
        0: Obstacles,
        1: Goal,
        2: Droplet,
        3: Health
        """
        obs = np.zeros_like(self.default_observation)
        # Goal layer
        x0,y0,x1,y1 = self.goal
        obs[y0:y1+1,x0:x1+1,1] = 1
        # Droplet layer
        x0,y0,x1,y1 = self.droplet
        obs[y0:y1+1,x0:x1+1,2] = 1
        # Health layer
        obs[:,:,3] = self.m_health
        return obs


    def _isComplete(self):
        if np.array_equal(self.droplet, self.goal):
            return True
        else:
            return False

