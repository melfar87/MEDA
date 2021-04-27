import copy
import math
import queue
import random
import numpy as np
from PIL import Image
from enum import IntEnum

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Direction(IntEnum):
    N = 0  # North
    E = 1  # East
    S = 2  # South
    W = 3  # West


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

    def __init__(self, height, width, n_bits=2, b_random=False, n_modules=0,
                 b_degrade=False, per_degrade=0.1, b_use_dict=False):
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

        # Instance variables
        assert height > 0 and width > 0
        self.height = height
        self.width = width
        self.n_bits = n_bits
        self.actions = Direction
        # Degradation parameters
        self.b_degrade = b_degrade
        self.m_taus = np.ones((height, width))
        self.m_C1s = np.zeros((height, width))
        self.m_C2s = np.zeros((height, width))
        # Degradation matrix
        self.m_degradation = np.ones((height, width))
        # Health matrix
        self.m_health = np.ones((height, width))
        # Actuations count matrix
        self.m_actuations_count = np.zeros((height, width))
        # Control pattern
        self.m_pattern = np.zeros((height, width))
        self.m_prev_pattern = np.zeros((height, width))
        # Number of steps 
        self.step_count = 0
        # Maximum number of steps
        self.max_step = 2 * (height + width)

        # Gym environment: action space
        self.action_space = spaces.Discrete(len(self.actions))
        if b_use_dict:
            self.observation_space = spaces.Dict({
                'health': spaces.Box(
                    low=0, high=2**n_bits-1,
                    shape=(height, width, 1), dtype=np.uint8),
                'sensors': spaces.Box(
                    low=0, high=1,
                    shape=(height, width), dtype=np.uint8)
            })
            self.keys = list(self.observation_space.spaces.keys())
        else:
            # Layers: (
                # 0: obstacles,
                # 1: goal,
                # 2: sensors,
                # 3: health )
            self.n_layers = 3 + 1
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(height, width, self.n_layers), dtype=np.float)
            self.default_observation = np.zeros(
                shape=(height, width, self.n_layers), dtype=np.float)
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
        # Reset steps counter
        self.step_count = 0
        # Reset actuations matrix
        self._resetActuationMatrix()
        # Get new start and goal locations and reset initial state
        self._resetInitialState()
        # Reset actuation pattern
        self.m_pattern[:,:] = 0
        self.m_prev_pattern[:,:] = 0
        # Update health
        self._updateHealth()
        self.m_distance = self._getDistanceToGoal()
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
        if self._isComplete():
            reward = 1.0
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
        return obs, reward, done, {}

    def render(self, mode='human'):
        """ Show environment """
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
        self.droplet = np.array((3,2,6,5))
        self.goal = np.array((6,9,9,12))
        self.agt_sta = copy.deepcopy(self.droplet)
        return
    
    def _resetActuationMatrix(self):
        """Resets actuation matrix
        """
        self.m_actuations_count = np.zeros((self.height, self.width))
        return
    
    def _updatePattern(self, action):
        """Update actuation pattern
        """
        x0,y0,x1,y1 = self.droplet
        # Find the shift in position
        if   action == Direction.N and self.droplet[3] < self.height:
            tmpShift = [ 0,+1, 0,+1]
            tmpFront = self.m_degradation[x0:x1,y1]
        elif action == Direction.S and self.droplet[1] > 0:
            tmpShift = [ 0,-1, 0,-1]
            tmpFront = self.m_degradation[x0:x1,y0-1]
        elif action == Direction.E and self.droplet[2] < self.width:
            tmpShift = [+1, 0,+1, 0]
            tmpFront = self.m_degradation[x1,y0:y1]
        elif action == Direction.W and self.droplet[0] > 0:
            tmpShift = [-1, 0,-1, 0]
            tmpFront = self.m_degradation[x0-1,y0:y1]
        else:
            tmpShift = [ 0, 0, 0, 0]
            tmpFront = [1,]
        # Update pattern
        tmpDr = self.droplet + tmpShift # New droplet location
        self.m_prev_pattern = np.copy(self.m_pattern)
        self.m_pattern[:,:] = 0 # Reset pattern
        self.m_pattern[tmpDr[0]:tmpDr[2],tmpDr[1]:tmpDr[3]] = 1 # New pattern
        # Update position
        if self.b_degrade:
            prob = np.mean(tmpFront)
            # if self.m_health[next_p[0]][next_p[1]] < 0.2:
            #    prob = 0.2
            # elif self.m_health[next_p[0]][next_p[1]] < 0.5:
            #    prob = 0.5
            # elif self.m_health[next_p[0]][next_p[1]] < 0.7:
            #    prob = 0.7
            # else:
            #    prob = 1.0
        else:
            prob = 1
        # Draw a sample from the bernolli distribution Bern(prob)
        if random.random() <= prob:
            # Random sample is True
            self.droplet += tmpShift
        else:
            # Random sample is False
            pass
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
            (np.ceil(self.m_degradation*(2**self.n_bits))-1)/(2**self.n_bits) )
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
        return obs

    def _isComplete(self):
        if np.array_equal(self.droplet, self.goal):
            return True
        else:
            return False

