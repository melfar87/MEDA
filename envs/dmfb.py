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
    N = 0 #North
    E = 1 #East
    S = 2 #South
    W = 3 #West

class Module:
    # basically a bbox in the DMFB
    def __init__(self, x_min, x_max, y_min, y_max):
        if x_min > x_max or y_min > y_max:
            raise TypeError('Module() inputs are illegal')
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def isPointInside(self, point):
        ''' point is in the form of (y, x) '''
        if point[0] >= self.y_min and point[0] <= self.y_max and\
                point[1] >= self.x_min and point[1] <= self.x_max:
            return True
        else:
            return False

    def isModuleOverlap(self, m):
        if self._isLinesOverlap(self.x_min, self.x_max, m.x_min, m.x_max) and\
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

class DMFBEnv(gym.Env):
    """ A digital microfluidic biochip environment
        [0,0]
          +---length---+-> x
          width       |
          +-------+
          |     [1,2]
          V
          y
    """
    metadata = {'render.modes':
            ['human', 'rgb_array']}
    def __init__(self, height, width, b_random = False,
            n_modules = 0, b_degrade = False,
            per_degrade = 0.1):
        super(DMFBEnv, self).__init__()
        assert width > 0 and height > 0
        self.width = height
        self.length = width
        self.actions = Direction
        self.action_space =\
                spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
                low = 0,
                high = 1,
                shape = (height, width, 3),
                dtype = 'uint8')
        self.reward_range = (-1.0, 1.0)
        self.b_random = b_random
        self.b_degrade = b_degrade
        self.max_step = 2 * (height + width)
        self.m_health = np.ones((height, width))
        self.m_usage = np.zeros((height, width))
        self.m_degrade = np.random.rand(height, width)
        self.m_degrade = self.m_degrade * 0.4 + 0.6
        selection = np.random.rand(height, width)
        per_healthy = 1. - per_degrade
        self.m_degrade[selection < per_healthy]\
                = 1.0
        self.step_count = 0
        if b_random:
            self.agt_pos, self.agt_end =\
                    self._randomSartNEnd()
        else:
            self.agt_pos = (0, 0)
            self.agt_end = (0, 1)
        self.agt_sta = copy.deepcopy(self.agt_pos)
        self.modules =\
                self._genRandomModules(n_modules)
        self.m_distance = self._computeDist()

    def step(self, action):
        done = False
        self.step_count += 1
        prev_dist = self._getDist()
        self._updatePosition(action)
        curr_dist = self._getDist()
        obs = self._get_obs()
        if self._isComplete():
            reward = 1.0
            done = True
        elif self.step_count > self.max_step:
            reward = -0.8
            done = True
        elif prev_dist > curr_dist: # move toward the goal
            reward = 0.5
        elif prev_dist == curr_dist:
            reward = -0.3
        else: # move away the goal
            reward = -0.8
        return obs, reward, done, {}

    def reset(self):
        self.step_count = 0
        if self.b_random is True:
            self.agt_pos, self.agt_end =\
                    self._randomSartNEnd()
        else:
            self.agt_pos, self.agt_end =\
                    self._getNextSartNEnd()
        self.agt_sta = copy.deepcopy(self.agt_pos)
        if len(self.modules) > 0:
            self.modules = self._genRandomModules()
        self._updateHealth()
        self.m_distance = self._computeDist()
        obs = self._get_obs()
        return obs

    def render(self, mode = 'human'):
        """ Show environment """
        #goal:2, pos:1, blocks:-1, degrade: -2
        if mode == 'human':
            img = np.zeros(shape =\
                    (self.width, self.length))
            img[self.agt_end[0]][self.agt_end[1]] = 2
            img[self.agt_pos[0]][self.agt_pos[1]] = 1
            for m in self.modules:
                for y in range(m.y_min, m.y_max + 1):
                    for x in range(
                            m.x_min, m.x_max + 1):
                        img[y][x] = -1
            if self.b_degrade:
                img[self.m_health < 0.5] = -2
            return img
        elif mode == 'rgb_array':
            img = self._get_obs().astype(np.uint8)
            for y in range(self.width):
                for x in range(self.length):
                    if np.array_equal(img[y][x], [1, 0, 0]): #red
                        img[y][x] = [255, 0, 0]
                    elif np.array_equal(img[y][x], [0,1,0]): #gre
                        img[y][x] = [0, 255, 0]
                    elif np.array_equal(img[y][x], [0,0,1]): #blu
                        img[y][x] = [0, 0, 255]
                    elif self.b_degrade and\
                            self.m_health[y][x] < 0.5: #ppl
                        img[y][x] = [255, 102, 255]
                    elif self.b_degrade and\
                            self.m_health[y][x] < 0.7: #ppl
                        img[y][x] = [255, 153, 255]
                    else: # grey
                        img[y][x] = [192, 192, 192]
            return img
        else:
            raise RuntimeError(
                    'Unknown mode in render')

    def close(self):
        """ close render view """
        pass

    def _genRandomModules(self, n_modules = 1):
        """ Generate reandom modules up to n_modules"""
        if self.width < 5 or self.length < 5:
            return []
        if n_modules * 4 / (self.width * self.length) > 0.2:
            print('Too many required modules in the environment.')
            return []
        modules = []
        for i in range(n_modules):
            x = random.randrange(0, self.length - 1)
            y = random.randrange(0, self.width - 1)
            m = Module(x, x+1, y, y+1)
            while m.isPointInside(self.agt_pos) or\
                    m.isPointInside(self.agt_end) or\
                    self._isModuleoverlap(m, modules):
                x = random.randrange(0, self.length - 1)
                y = random.randrange(0, self.width - 1)
                m = Module(x, x+1, y, y+1)
            modules.append(m)
        return modules

    def printHealthSatus(self):
        print('### Env Health ###')
        n_bad = np.count_nonzero(self.m_health < 0.2)
        n_mid = np.count_nonzero(self.m_health < 0.5)
        n_ok = np.count_nonzero(self.m_health < 0.8)
        print('Really bad:', n_bad,
                'Halfly degraded:', n_mid - n_bad,
                'Mildly degraded', n_ok - n_mid)

    def _isModuleoverlap(self, m, modules):
        for mdl in modules:
            if mdl.isModuleOverlap(m):
                return True
        return False

    def _computeDist(self):
        m_dist = np.zeros(
                shape = (self.width, self.length),
                dtype = np.uint8)
        q = queue.Queue()
        q.put(self.agt_end)
        m_dist[self.agt_end[0]][self.agt_end[1]] = 1
        self._setModulesWithValue(m_dist, np.iinfo(np.uint8).max)
        while not q.empty():
            q, m_dist = self._updateQueue(q, m_dist)
        return m_dist

    def _setModulesWithValue(self, m_dist, v):
        for m in self.modules:
            for x in range(m.x_min, m.x_max + 1):
                for y in range(m.y_min, m.y_max + 1):
                    m_dist[y][x] = v
        return

    def _updateQueue(self, q, m_dist):
        head = q.get()
        dist = m_dist[head[0]][head[1]]
        neighbors = self._getNeighbors(head)
        for n in neighbors:
            if m_dist[n[0]][n[1]] == 0:
                q.put(n)
                m_dist[n[0]][n[1]] = dist + 1
        return q, m_dist

    def _getNeighbors(self, p):
        neighbors = [
                (p[0] - 1, p[1]),
                (p[0] + 1, p[1]),
                (p[0], p[1] - 1),
                (p[0], p[1] + 1)]
        return [n for n in neighbors if self._isPointInside(n)]

    def _randomSartNEnd(self):
        x = random.randrange(0, self.length)
        y = random.randrange(0, self.width)
        start = (y, x)
        repeat = random.randrange(0, self.length * self.width)
        for i in range(repeate):
            x = random.randrange(0, self.length)
            y = random.randrange(0, self.width)
        end = (y, x)
        while end == start:
            x = random.randrange(0, self.length)
            y = random.randrange(0, self.width)
            end = (y, x)
        return start, end

    def _getNextSartNEnd(self):
        start = self.agt_sta
        end = self.agt_end
        end = self._getRightPoint(end)
        if start == end:
            start = self._getRightPoint(start)
            end = self._getRightPoint(start)
        return start, end

    def _getRightPoint(self, point):
        y = point[0]
        x = point[1] + 1
        if x == self.length:
            x = 0
            y += 1
            if y == self.width:
                y = 0
        return (y, x)

    def _getDist(self):
        y = self.agt_pos[0]
        x = self.agt_pos[1]
        return self.m_distance[y][x]

    def _updatePosition(self, action):
        # update self.agt_pos
        next_p = list(self.agt_pos)
        # Not moving if stuck in a bad electrode
        if self.b_degrade:
            prob = self.m_health[next_p[0]][next_p[1]]
            #if self.m_health[next_p[0]][next_p[1]] < 0.2:
            #    prob = 0.2
            #elif self.m_health[next_p[0]][next_p[1]] < 0.5:
            #    prob = 0.5
            #elif self.m_health[next_p[0]][next_p[1]] < 0.7:
            #    prob = 0.7
            #else:
            #    prob = 1.0
            if random.random() > prob:
                return# got random stuck
        if action == Direction.N:
            next_p[0] -= 1
        elif action == Direction.E:
            next_p[1] += 1
        elif action == Direction.S:
            next_p[0] += 1
        else: # Direction.W
            next_p[1] -= 1
        if not self._isPointInside(next_p):
            return # no update
        elif self._isTouchingModule(next_p):
            return # no update
        else: # legal position
            self.agt_pos = tuple(next_p)
            if self.b_degrade:
                self.m_usage[next_p[0]][next_p[1]] += 1
            return

    def _isPointInside(self, point):
        if point[1] < 0 or point[1] >= self.length:
            return False
        if point[0] < 0 or point[0] >= self.width:
            return False
        return True

    def _isTouchingModule(self, point):
        for m in self.modules:
            if point[1] >= m.x_min and\
                    point[1] <= m.x_max and\
                    point[0] >= m.y_min and\
                    point[0] <= m.y_max:
                return True
        return False

    def _isComplete(self):
        if self.agt_pos == self.agt_end:
            return True
        else:
            return False

    def _get_obs(self):
        """
        RGB format of image
        Obstacles - red in layer 0
        Goal      - greed in layer 1
        Droplet   - blue in layer 2
        """
        obs = np.zeros(
                shape = (self.width, self.length, 3))
        obs = self._addModulesInObs(obs)
        obs[self.agt_end[0]][self.agt_end[1]][1] = 1
        obs[self.agt_pos[0]][self.agt_pos[1]][2] = 1
        return obs

    def _updateHealth(self):
        if not self.b_degrade:
            return
        n_unhealthy = np.count_nonzero(self.m_health < 0.1)
        if n_unhealthy > 2:
            return
        index = self.m_usage > 50.0
        self.m_health[index] = self.m_health[index] *\
                self.m_degrade[index]
        self.m_usage[index] = 0

    def _addModulesInObs(self, obs):
        for m in self.modules:
            for y in range(m.y_min, m.y_max + 1):
                for x in range(m.x_min, m.x_max + 1):
                    obs[y][x][0] = 1
        return obs
