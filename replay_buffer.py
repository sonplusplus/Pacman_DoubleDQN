# replay_buffer.py
import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Bộ nhớ đệm để lưu trữ và lấy mẫu kinh nghiệm cho DQN
    """
    def __init__(self, capacity):
        """
        Khởi tạo replay buffer
        
        Args:
            capacity: kích thước tối đa của buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Thêm một transition vào buffer
        
        Args:
            state: trạng thái hiện tại
            action: hành động đã thực hiện
            reward: phần thưởng nhận được
            next_state: trạng thái tiếp theo
            done: trạng thái kết thúc hay chưa
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Lấy ngẫu nhiên một batch từ buffer
        
        Args:
            batch_size: kích thước batch
            
        Returns:
            tuple of (states, actions, rewards, next_states, dones)
        """
        # Kiểm tra kích thước buffer
        assert len(self.buffer) >= batch_size, "Buffer không đủ dữ liệu để lấy mẫu"
        
        # Lấy mẫu ngẫu nhiên
        batch = random.sample(self.buffer, batch_size)
        
        # Tách batch thành các thành phần
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển đổi sang numpy arrays
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states), 
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        """
        Lấy kích thước hiện tại của buffer
        
        Returns:
            int: số lượng transitions trong buffer
        """
        return len(self.buffer)