import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DoubleDQN(nn.Module):
    def __init__(self, input_shape,n_actions ): #possible actions
        super(DoubleDQN, self).__init__()
        self.conv = nn.Sequential(
            #Convolution 1
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            #Convolution 2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            #Convolution 3
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
    def _get_conv_out(self, shape):
        #size of cnn output
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()[1:]))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
        