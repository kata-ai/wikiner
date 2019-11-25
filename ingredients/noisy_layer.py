import math

import torch
from torch import nn
from torch.nn import Parameter

class Channel(nn.Module):
    def __init__(self, input_dim, label_confusion, bias = False):
        super(Channel, self).__init__()        
        self.input_dim = input_dim
        self.activation = nn.Softmax(dim=1)
    
        # use the label_confusion as weights             
        self.weight = Parameter(label_confusion)
        if bias:
            self.bias = Parameter(torch.zeros(input_dim))  
        else:
            self.register_parameter('bias', None)   
        
    def forward(self, x):
        channel_matrix = self.activation(self.weight)        
        return torch.matmul(x, channel_matrix)
    
    def reset_parameters(self):
        n = self.input_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
