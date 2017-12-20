# Import libraries
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

# CONNECT TO DATABASE
conn = sqlite3.connect('kassandr.db')
c = conn.cursor()

nv = 2158860        # total # of offers 
nh = 100
rbm = RBM(nv, nh)

offer_id_counter = 2158860
nb_users = 291485
batch_size = 10000



training_set = []

# TRAIN
nb_epoch = 10
start = timer()
for epoch in range(1, nb_epoch + 1):
    start_epoch = timer()
    train_loss = 0
    s = 0.
    for current_user in range(0, nb_users):
        if not current_user % batch_size:
            print("Training for user: " + str(current_user))
            current_user_training_set = []
            for row in c.execute("SELECT * FROM ratings WHERE user_id > " + str(current_user) + " AND user_id < " + 
                                 str(current_user + batch_size) + ";"):
                current_user_training_set.append(row)
            current_user_ratings = np.full(offer_id_counter, -1.0)
            for i in range(0, len(current_user_training_set)):
                if current_user_training_set[i][2] != "nan":
                    current_user_ratings[int(current_user_training_set[i][1])] = current_user_training_set[i][2]
            training_set.append(current_user_ratings)
            training_set_torch = torch.FloatTensor(training_set)
            vk = training_set_torch
            v0 = training_set_torch
            ph0, _ = rbm.sample_h(v0)
            for k in range(20):         # random walk
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0<0] = v0[v0<0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
            s += 1.
            # reset variables
            training_set = []
            current_user_ratings = []
            current_user_training_set = []
    print('epoch: ' + str(epoch) + ', train loss: ' + str(train_loss/s))
    end_epoch = timer()
    print("Epoch time elapsed")
    print(end_epoch - start_epoch)
end = timer()
print("Time elapsed")
print(end - start)

# TEST
test_set = []
training_set = []
test_loss = 0
s = 0.
start = timer()
for current_user in range(1, nb_users + 1):
    if not current_user % batch_size:
        print("Testing for user: " + str(current_user))
        # get test set of user
        current_user_test_set = []
        for row in c.execute("SELECT * FROM ratings_test WHERE user_id > " + str(current_user) + " AND user_id < " +
                             str(current_user + batch_size) + ";"):
            current_user_test_set.append(row)
        current_user_test_ratings = np.full(offer_id_counter, -1.0)
        for i in range(0, len(current_user_test_set)):
            if current_user_test_set[i][2] != "nan":
                current_user_test_ratings[int(current_user_test_set[i][1])] = current_user_test_set[i][2]
        test_set.append(current_user_test_ratings)
        test_set_torch = torch.FloatTensor(test_set)
        
        # get training set of user
        current_user_train_set = []
        for row in c.execute("SELECT * FROM ratings WHERE user_id > " + str(current_user) + " AND user_id < " + 
                                 str(current_user + batch_size) + ";"):
            current_user_training_set.append(row)
        current_user_ratings = np.full(offer_id_counter, -1.0)
        for i in range(0, len(current_user_training_set)):
            if current_user_training_set[i][2] != "nan":
                current_user_ratings[int(current_user_training_set[i][1])] = current_user_training_set[i][2]
        training_set.append(current_user_ratings)
        training_set_torch = torch.FloatTensor(training_set)
        
        v = training_set_torch
        vt = test_set_torch
        if len(vt[vt>=0]) > 0:
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)
            test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
            s += 1.
print ("Test loss: " + str(test_loss/s))
end = timer()
print("Time elapsed")
print(end - start)
        
        
    
        
