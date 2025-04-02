import torch
import torch.nn.functional as F
import numpy as np
import random
from double_dqn.model import DoubleDQN

class DoubleDQNA:
    def __init__(self, state_dim, action_dim, device="cpu", config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.config = config if config is not None else {}
        self.lr = self.config.get("lr", 0.001)
        self.gamma = self.config.get("gamma", 0.99)
        self.epsilon = self.config.get("epsilon", 1.0)
        self.epsilon_min = self.config.get("epsilon_min", 0.01)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 64)
        self.target_update_interval = self.config.get("target_update_interval", 10)
        
        #start network
        self.online_network = DoubleDQN(state_dim, action_dim).to(device)
        self.target_network = DoubleDQN(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimize = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        
        self.count = 0

    def get_action(self, state, evaluate=False):
        #random movement
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            #best move
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.online_network(state)
            action = q_values.argmax(dim=1).item()
        return action
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        #q values
        q_vals = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_move = torch.argmax(self.online_network(next_states), dim=1)
            next_q_vals = self.target_network(next_states).gather(1, next_move.unsqueeze(1)).squeeze(1)
            target_q_vals = rewards + self.gamma * (1 - dones.float()) * next_q_vals
        
        #loss
        loss = F.smooth_l1_loss(q_vals, target_q_vals)
        
        #optimize
        self.optimize.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10)
        self.optimize.step()
        self.count += 1

        if self.count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return loss.item()