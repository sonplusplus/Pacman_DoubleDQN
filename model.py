# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNModel(nn.Module):
    """
    Mạng neural sử dụng trong Deep Q-Learning
    """
    def __init__(self, input_shape, n_actions):
        """
        Khởi tạo mạng DQN
        
        Args:
            input_shape: tuple (channels, height, width)
            n_actions: số lượng actions
        """
        super(DQNModel, self).__init__()
        
        # Lớp Convolutional
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Tính kích thước output của các lớp conv
        conv_out_size = self._get_conv_output(input_shape)
        
        # Lớp fully connected
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        """
        Tính kích thước output của các lớp convolution
        
        Args:
            shape: kích thước đầu vào (channels, height, width)
            
        Returns:
            int: kích thước output flatten
        """
        batch = torch.zeros(1, *shape)
        conv_out = self.conv(batch)
        return int(np.prod(conv_out.size()))
    
    def forward(self, x):
        """
        Forward pass của mạng neural
        
        Args:
            x: tensor đầu vào, shape (batch, channels, height, width)
               với giá trị pixel trong khoảng [0, 255]
               
        Returns:
            tensor Q-values cho mỗi action
        """
        # Normalize input từ [0, 255] sang [0, 1]
        x = x.float() / 255.0
        
        # Đưa qua các lớp convolution
        conv_out = self.conv(x)
        
        # Flatten
        conv_out = conv_out.view(x.size()[0], -1)
        
        # Đưa qua các lớp fully connected
        return self.fc(conv_out)