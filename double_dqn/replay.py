import numpy as np
import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next, done):
        self.buffer.append((state, action, reward, next, done))
        #add cur model's state
    
    #sample random
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    #size of buffer
    def __len__(self):
        return len(self.buffer)