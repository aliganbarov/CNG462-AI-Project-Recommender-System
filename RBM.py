import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from timeit import default_timer as timer
import sqlite3

# RBM class
class RBM:
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)            # init matrix nh x nv, with normal distribution with mean 0 and var 1
        self.a = torch.randn(1, nh)             # bias for hidden nodes, frst dim is batch, second dim is bias
        self.b = torch.randn(1, nv)             # bias for visible nodes
    
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)  # create activation function by adding wx and a (expand_as adds a to whole wx)
        p_h_given_v = torch.sigmoid(activation) # probability h is activated given v / sigmoid function of activation
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)  # create activation function by adding wx and a (expand_as adds a to whole wx)
        p_v_given_h = torch.sigmoid(activation) # probability v is activated given h / sigmoid function of activation
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)