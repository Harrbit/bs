import torch
import torch.nn as nn


class ValueNet(nn.modules):
    def __init__(self, state_dim, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
    

class ActionNet(nn.modules):
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)
    
class TRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, critic_lr, gamma):
        self.actor = ActionNet(state_dim,hidden_dim , action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)
        self.critic_optmzr = torch.optim.Adam(self.critic.parameters(), 
                                              lr = critic_lr)
        self.gamma = gamma

    def take_action(self, state):
        action = self.actor(state)
        return action
    
    def update(self, trasition_dict):
        