#!/usr/bin/python


from envs.meda import*

class OldRouter:
    """ An old router for DMFBs """
    def __init__(self, env):
        self.width = env.height
        self.length = env.width
        self.start = env.agt_sta
        self.end = env.goal
        #self.modules = env.modules
        self.m_dist = self._getDistanceToGoal()
        self.b_degrade = env.b_degrade
        self.m_health = env.m_health
    
    def getReward(self, b_path=False):
        curr_dist = self._getDistanceToGoal()
        if curr_dist == 0:
            reward = 1.0
        else:  # move toward the goal
            reward = -1.0 * curr_dist
        return reward
    
    def _getDistanceToGoal(self):
        dist = np.mean(np.abs(self.end - self.start))
        return dist
    
    def _isComplete(self):
        if np.array_equal(self.start, self.end):
            return True
        else:
            return False
