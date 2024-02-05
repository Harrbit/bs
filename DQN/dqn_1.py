import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNET(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, 
                 gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = QNET(state_dim, hidden_dim, action_dim).to(device)
        self.target__net = QNET(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = learning_rate)
        self.epsilon = epsilon
        self.device = device

    def take_action(self, state):
        if np.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state],dtype=float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'], 
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']
                              ).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                                dtype = torch.float).to(self.device)
        next_state = torch.tensor(transition_dict[next_state],
                                   dtype=torch.float).view(-1,1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], 
                             dtype=torch.float).to(self.device)
        
        q_values = self.q_net(states).gather(1,actions)
        max_next_q_values = self.target__net(next_state).max([1])[0].view(-1,1)
        
