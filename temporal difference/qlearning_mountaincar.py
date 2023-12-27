import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import gym
import time
import matplotlib

class EGreedyExpStrategy(): # epsilons-greedy strategy
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=1000000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, q_values, state):

        if np.random.rand() > self.epsilon:
            action = np.argmax(q_values)
            self.exploratory_action_taken = False
        else:
            action = np.random.randint(len(q_values))
            self.exploratory_action_taken = True

        self._epsilon_update()
        return action
    
def get_discrete_state(state): # get the state from environment and transfer the observation into integer
    DISCRETE_OS_SIZE = [20,20]
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # Determine the size of state space
    num_states = [20, 20]
    
    # randomly initialize a Q table
    Q = np.random.uniform(low=-2, high=0, size=(num_states + [env.action_space.n]))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    reach_rate = np.zeros(int(episodes/100))
    r = 0
    # Calculate episodic reduction in epsilon
    training_strategy = EGreedyExpStrategy(init_epsilon=0.8, min_epsilon=0.01, decay_steps=500000)
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state
        state_adj = get_discrete_state(state)
        while done != True:               
            # Determine next action - epsilon greedy strategy
            action = training_strategy.select_action(Q[state_adj[0], state_adj[1]], state)
                
            # Get next state and reward
            new_state, reward, done, info = env.step(action) 
            
                
            # Discretize new_state
            new_state_adj = get_discrete_state(new_state)
            #Allow for terminal states
            if done and new_state[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[new_state_adj[0], 
                                                   new_state_adj[1]]) - 
                                 Q[state_adj[0], state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = new_state_adj
        if tot_reward != -200.0:
                reach_rate[r] = reach_rate[r] + 1
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = [] 
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            print(reach_rate[r])
            r = r + 1
    env.close()
    
    return ave_reward_list, reach_rate

# Run Q-learning algorithm
rewards, reach = QLearning(env, 0.2, 0.9, 0.8, 0, 7500)

plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.plot(100*(np.arange(len(reach)) + 1), reach)
plt.xlabel('Episodes')
plt.ylabel('Average Reward & Reach times')
plt.title('Average Reward & Reach times vs Episodes')
plt.savefig('rewards.jpg')     
plt.close()  

