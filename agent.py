# agent.py
import torch
import torch.nn.functional as F
import numpy as np
import random

from model import DQNModel
from replay_buffer import ReplayBuffer

class DoubleDQNAgent:
    """
    Agent sử dụng Double Deep Q-Network
    """
    def __init__(
        self,
        state_shape,
        n_actions,
        device,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=10**5,
        buffer_size=10**5,
        batch_size=32,
        target_update=1000
    ):
        """
        Khởi tạo Double DQN Agent
        
        Args:
            state_shape: tuple kích thước state (channels, height, width)
            n_actions: số lượng actions
            device: thiết bị xử lý (cpu/cuda)
            learning_rate: tốc độ học
            gamma: hệ số discount
            epsilon_start: epsilon ban đầu cho exploration
            epsilon_final: epsilon cuối cùng
            epsilon_decay: tốc độ giảm của epsilon
            buffer_size: kích thước replay buffer
            batch_size: kích thước batch
            target_update: số bước để cập nhật target network
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.policy_net = DQNModel(state_shape, n_actions).to(device)
        self.target_net = DQNModel(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network luôn ở chế độ evaluation
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Counter
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Chọn action dựa trên chính sách epsilon-greedy
        
        Args:
            state: state hiện tại (numpy array)
            training: có đang trong quá trình huấn luyện hay không
            
        Returns:
            int: action được chọn
        """
        if training:
            # Cập nhật epsilon
            self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                          np.exp(-self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            
            # Exploration
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def optimize(self):
        """
        Huấn luyện mạng neural với một batch từ replay buffer
        
        Returns:
            float: loss hoặc None nếu buffer chưa đủ dữ liệu
        """
        # Kiểm tra xem buffer có đủ dữ liệu không
        if len(self.buffer) < self.batch_size:
            return None
        
        # Lấy batch từ replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Chuyển sang tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Tính current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: chọn actions bằng policy network
        next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
        
        # Tính giá trị Q cho next actions bằng target network
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        # Mask cho các states terminal
        next_q_values = next_q_values * (1 - dones)
        
        # Tính expected Q values
        expected_q_values = rewards + self.gamma * next_q_values
        
        # Tính loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Cập nhật target network với trọng số từ policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """
        Lưu model
        
        Args:
            path: đường dẫn để lưu model
        """
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        """
        Tải model
        
        Args:
            path: đường dẫn đến file model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']