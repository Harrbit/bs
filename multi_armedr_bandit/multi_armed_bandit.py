import numpy as np
import matplotlib.pyplot as plt
'''
BernoulliBandit类：包含实例中每个老虎机的中奖概率和一些相关信息，其中的step方法用来判断是否中奖
'''


class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 这里给出了所有老虎机的中奖概率
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K
    
    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)
K = 10
bandit_10_arms = BernoulliBandit(K)  # 将BernoulliBandit类实例化
# print("%d,%.4f" % (bandit_10_arms.best_idx, bandit_10_arms.best_prob))

'''
solver类记录了在反复尝试过程中的每一步动作以及这些动作带来的regret，以及regret的积累
存在一个问题：NotImplementedError在这里是怎么工作的，不太懂，似乎是在solver中留一个占位符，等着被子类中的同名函数重载？
'''


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.action = []
        self.regrets = []

    def update_regret(self, k):
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


'''epsilon—greedy,solver的子类，这里的run_one_step实现了epsilon的随机挑选'''


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k] + 1) * (r - self.estimates[k])
        return k


'''画图函数，没什么说的'''


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arms, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print(epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["epsilon greedy"])

# 展示不同epsilon下的regret
np.random.seed(0)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arms, epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon = {}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
