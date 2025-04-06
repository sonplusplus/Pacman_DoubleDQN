import numpy as np
import random
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import DQN

class DoubleDQNAgent:
    def __init__(self, env, config):
        """
        Initialize Double DQN Agent
        
        Args:
            env: Gym environment
            config: Configuration dictionary with hyperparameters
        """
        self.env = env
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        # Hyperparameters
        self.gamma = config['gamma']  # Discount factor
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.buffer_size = config['buffer_size']
        self.epsilon = config['epsilon_start']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.target_update = config['target_update']
        self.device = config['device']
        self.current_step = 0
        
        # Initialize replay buffer
        self.memory = deque(maxlen=self.buffer_size)
        
        # Neural network for Q-value approximation
        in_channels = self.state_dim[0]
        self.policy_net = DQN(in_channels=in_channels, n_actions=self.action_dim).to(self.device)
        self.target_net = DQN(in_channels=in_channels, n_actions=self.action_dim).to(self.device)
        
        # Copy policy network parameters to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
    def preprocess_state(self, state):
        """
        Preprocess the state before passing it to the network
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed state tensor
        """
        # Convert to float and normalize
        state = state.astype(np.float32) / 255.0  
        
        # Convert to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return state
        
    def select_action(self, state, evaluation=False):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            evaluation: If True, use greedy policy
            
        Returns:
            Selected action
        """
        if (random.random() > self.epsilon) or evaluation:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
            
        # Decay epsilon
        if not evaluation and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """
        Update policy network parameters
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.cat(states)
        action_batch = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_batch = torch.cat(next_states)
        done_batch = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Double DQN: Use policy network to select action and target network to evaluate action
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            
            # Get Q-values from target network for next actions
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            
            # Compute the expected Q values
            expected_q_values = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10, 10)
            
        self.optimizer.step()
        
        # Update target network
        self.current_step += 1
        if self.current_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
    
    def save_model(self, path):
        """
        Save model parameters
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'current_step': self.current_step
        }, path)
        
    def load_model(self, path):
        """
        Load model parameters
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.current_step = checkpoint['current_step']