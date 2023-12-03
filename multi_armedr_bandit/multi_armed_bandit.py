import numpy as np
'''理想状态下的多臂老虎机问题解决方法'''
class BernouliBandit:
    def __init__(self,k):
        self.prob = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.prob)
        self.best_prob = self.prob[self.best_idx]
        self.k = k
    
    def step(self, k):
        if np.random.rand < self.prob[k]:
            return 1
        else:
            return 0

np.random.seed(1)
k = 1000
bandit_10_arms=BernouliBandit(k)
print("%d,%.4f"%(bandit_10_arms.best_idx,bandit_10_arms.best_prob))