import numpy as np
'''理想状态下的多臂老虎机问题解决方法'''


class BernouliBandit:
    def __init__(self, k):
        self.prob = np.random.uniform(size=k)
        self.best_idx = np.argmax(self.prob)
        self.best_prob = self.prob[self.best_idx]
        self.k = k
    
    def step(self, k):
        if np.random.rand < self.prob[k]:
            return 1
        else:
            return 0

'''
np.random.seed(1)
K = 10000
bandit_10_arms = BernouliBandit(K)
print("%d,%.4f" % (bandit_10_arms.best_idx, bandit_10_arms.best_prob))
'''
class solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.action = []
        self.regrets = []

    def update_regret(self,k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.action.append(k)
            self.update_regret(k)
