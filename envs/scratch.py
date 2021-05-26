# import stable_baselines

# import matplotlib
# import matplotlib.pyplot as plt
# import tensorflow as tf

# from utils import OldRouter
# from my_net import MyCnnPolicy
# from envs.dmfb import *
# from envs.meda import *

# from stable_baselines.common import make_vec_env, tf_util
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import PPO2

class SomeModule:
    
        #return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    # goal:2, pos:1, blocks:-1, degrade: -2
    # if mode == 'human':
    #     img = np.zeros(shape= \
    #                        (self.height, self.width))
    #     img[self.goal[0]][self.goal[1]] = 2
    #     img[self.droplet[0]][self.droplet[1]] = 1
    #     for m in self.modules:
    #         for y in range(m.y_min, m.y_max + 1):
    #             for x in range(
    #                     m.x_min, m.x_max + 1):
    #                 img[y][x] = -1
    #     if self.b_degrade:
    #         img[self.m_health < 0.5] = -2
    #     return img
    # elif mode == 'rgb_array':
    #     img = self._getObs().astype(np.uint8)
    #     for y in range(self.height):
    #         for x in range(self.width):
    #             if np.array_equal(img[y][x], [1, 0, 0]):  # red
    #                 img[y][x] = [255, 0, 0]
    #             elif np.array_equal(img[y][x], [0, 1, 0]):  # gre
    #                 img[y][x] = [0, 255, 0]
    #             elif np.array_equal(img[y][x], [0, 0, 1]):  # blu
    #                 img[y][x] = [0, 0, 255]
    #             elif self.b_degrade and \
    #                     self.m_health[y][x] < 0.5:  # ppl
    #                 img[y][x] = [255, 102, 255]
    #             elif self.b_degrade and \
    #                     self.m_health[y][x] < 0.7:  # ppl
    #                 img[y][x] = [255, 153, 255]
    #             else:  # grey
    #                 img[y][x] = [192, 192, 192]
    #     return img
    # else:
    #     raise RuntimeError(
    #         'Unknown mode in render')
        

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
            shape=(self.height, self.width),
            dtype=np.uint8)
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
    

#!/usr/bin/python

import queue
from envs.dmfb import*

class OldRouter:
    """ An old router for DMFBs """
    def __init__(self, env):
        self.width = env.height
        self.length = env.width
        self.start = env.agt_sta
        self.end = env.goal
        #self.modules = env.modules
        self.m_dist = env._getDistanceToGoal()
        self.b_degrade = env.b_degrade
        self.m_health = env.m_health

    def getReward(self, b_path = False):
        path = [self.start]
        dist = self.m_dist[self.start[0]][self.start[1]]
        while dist > 1:
            old_dist = dist
            neighbors = self._getNeighbors(path[-1])
            for n in neighbors:
                if self.m_dist[n[0]][n[1]] ==\
                        dist - 1:
                    path.append(n)
                    dist = dist - 1
                    break
            if old_dist == dist:
                print('something wrong in getReward')
                print(path)
                print(self.m_dist)
                break
        if not self.b_degrade:
            reward = (len(path)-2) * (0.5) + 1.0
            if b_path:
                return len(path) - 1
            else:
                return reward
        else:
            num_steps = 0.
            for step in path[:-1]:
                prob = self.m_health[step[0]][step[1]]
                num_steps += 1. / prob
            reward = (len(path) - 2) * 0.5 + 1.0
            reward += (num_steps - len(path) + 2) * (-0.3)
            if b_path:
                return num_steps
            else:
                return reward

    def _computeDist(self):
        m_dist = np.zeros(
                shape = (self.width, self.length),
                dtype = np.uint8)
        q = queue.Queue()
        q.put(self.end)
        m_dist[self.end[0]][self.end[1]] = 1
        self._setModulesWithValue(m_dist, np.iinfo(np.uint8).max)
        while not q.empty():
            q, m_dist = self._updateQueue(q, m_dist)
        return m_dist
    
    def _getDistanceToGoal(self):
        dist = np.mean(np.abs(self.goal - self.droplet))
        return dist

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
        return [n for n in neighbors if self._isWithinBoundary(n)]

    def _isWithinBoundary(self, p):
        if p[0] < self.width and p[0] >= 0 and\
                p[1] < self.length and p[1] >= 0:
            return True
        else:
            return False

