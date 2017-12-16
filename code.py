# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

user_id_counter = 0
offer_id_counter = 0

user_id_to_int_map = {}
offer_id_to_int_map = {}

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
        
        
def replace_index(data, my_dict, k):
    for i in range(0, data.shape[0]):
        if data[i] in my_dict:
            data[i] = my_dict[data[i]]
        else:
            my_dict[data[i]] = k
            data[i] = k
            k += 1
    return data, k



names = ["userid", "offerid", "countrycode", "category", "merchant", "utcdate", "rating"]

training_set = []

nv = 2158860        # total # of offers 
nh = 100
rbm = RBM(nv, nh)

offer_id_counter = 2158860

two_users = 0

for current_user in range(3, 100):
    two_users += 1
    print("Training for user: " + str(current_user))
    current_user_training_set = []
    chunksize = 10 ** 5
    # read from every training file
    for i in range(1, 17):
        # read files in chunks
        print("Reading train file: " + str(i))
        for training_chunk in pd.read_csv('train/train' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
#        for training_chunk in pd.read_csv('train_de3_replaced.csv', chunksize=chunksize, names=names):
            training_chunk = np.array(training_chunk, dtype='str')
#            training_chunk[:, 0], user_id_counter = replace_index(training_chunk[:, 0], user_id_to_int_map, user_id_counter)
#            training_chunk[:, 1], offer_id_counter = replace_index(training_chunk[:, 1], offer_id_to_int_map, offer_id_counter)
            for i in range(0, len(training_chunk[:, 0])):
                if int(training_chunk[:, 0][i]) == current_user:
                    current_user_training_set.append([training_chunk[:, 1][i], 
                                                     training_chunk[:, 6][i]])
    current_user_ratings = np.zeros(offer_id_counter)
    for i in range(0, len(current_user_training_set)):
        if current_user_training_set[i][1] != "nan":
            current_user_ratings[int(current_user_training_set[i][0])] = int(current_user_training_set[i][1])
    training_set.append(current_user_ratings)
    
    if two_users == 2:
        two_users = 0
        # Training part for current two users
        training_set_torch = torch.FloatTensor(training_set)
        
        # Training the RBM
        nb_epoch = 10
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            vk = training_set_torch
            v0 = training_set_torch
            ph0, _ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0 - vk))
            s += 1.
            print('epoch: ' + str(epoch) + ', train loss: ' + str(train_loss/s))
        
        # reset variables
        training_set = []

# TESTING
def replace_index_test(data, my_dict):
    for i in range(0, data.shape[0]):
        if data[i] in my_dict:
            data[i] = my_dict[data[i]]
    return data
two_users = 0
test_set = []
test_loss = 0
s = 0.
for current_user in range(1, 3):
    two_users += 1
    print("Testing for user: " + str(current_user))
    current_user_test_set = []
    chunksize = 10 ** 5
    # read from every test file
    for i in range(1, 3):
        print("Reading test file: " + str(i))
        for test_chunk in pd.read_csv('test/test' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
            test_chunk = np.array(test_chunk, dtype='str')
#            test_chunk[:, 0] = replace_index_test(test_chunk[:, 0], user_id_to_int_map)
#            test_chunk[:, 1] = replace_index_test(test_chunk[:, 1], offer_id_to_int_map)
            for i in range(0, len(test_chunk[:, 0])):
                try:
                    if int(test_chunk[:, 0][i]) == current_user:
                        current_user_test_set.append([test_chunk[:, 1][i], test_chunk[:, 6][i]])
                except ValueError:
                    pass
    current_user_ratings = np.zeros(offer_id_counter)
    for i in range(0, len(current_user_test_set)):
        if current_user_test_set[i][1] != "nan":
            try:
                current_user_ratings[int(current_user_test_set[i][0])] = int(current_user_test_set[i][1])
            except ValueError:
                pass
    test_set.append(current_user_ratings)
    
    if two_users == 2:
        two_users = 0
        # Testing part for current two users
        test_set_torch = torch.FloatTensor(test_set)
        
        # test
        v = test_set_torch
        vt = test_set_torch
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt - v))
        s += 1.
print('test loss: ' + str(test_loss/s))



# replace user_id, offer_id by integers 0..n
#training_set[:,0] = replace_index(training_set[:,0])
#training_set[:,1] = replace_index(training_set[:,1])
#training_set[:,2] = replace_index(training_set[:,2])
#training_set[:,3] = replace_index(training_set[:,3])
#training_set[:,4] = replace_index(training_set[:,4])
#training_set[:,5] = replace_index(training_set[:,5])
#test_set[:,0] = replace_index(test_set[:,0])
#test_set[:,1] = replace_index(test_set[:,1])
#test_set[:,2] = replace_index(test_set[:,2])
#test_set[:,3] = replace_index(test_set[:,3])
#test_set[:,4] = replace_index(test_set[:,4])
#test_set[:,5] = replace_index(test_set[:,5])
#
## convert data to int
#training_set = np.array(training_set, dtype='int')
#test_set = np.array(test_set, dtype='int')
#
## Number of users and offers
#nb_users = int(max(training_set[:,0]))
#nb_offers = int(max(training_set[:,1]))
#
## Converting the data into list of users containing list of offers with ratings
##training_set = convert(training_set)
##test_set = convert(test_set)
#
## training for user 1
#user_1_training_set = convert(training_set, 104)
#user_1_test = convert(test_set, 104)
#
#for i in range(0, len(user_1_test)):
#    if (user_1_test[i] == 1):
#        print("rated at " + str(i))
#
## Converting the data into Torch tensors
#training_set = torch.FloatTensor(training_set)
#test_set = torch.FloatTensor(test_set)
#
# Creating the architecture of the Neural Network
#class RBM:
#    def __init__(self, nv, nh):
#        self.W = torch.randn(nh, nv)            # init matrix nh x nv, with normal distribution with mean 0 and var 1
#        self.a = torch.randn(1, nh)             # bias for hidden nodes, frst dim is batch, second dim is bias
#        self.b = torch.randn(1, nv)             # bias for visible nodes
#    
#    def sample_h(self, x):
#        wx = torch.mm(x, self.W.t())
#        activation = wx + self.a.expand_as(wx)  # create activation function by adding wx and a (expand_as adds a to whole wx)
#        p_h_given_v = torch.sigmoid(activation) # probability h is activated given v / sigmoid function of activation
#        return p_h_given_v, torch.bernoulli(p_h_given_v)
#    
#    def sample_v(self, y):
#        wy = torch.mm(y, self.W)
#        activation = wy + self.b.expand_as(wy)  # create activation function by adding wx and a (expand_as adds a to whole wx)
#        p_v_given_h = torch.sigmoid(activation) # probability v is activated given h / sigmoid function of activation
#        return p_v_given_h, torch.bernoulli(p_v_given_h)
#    
#    def train(self, v0, vk, ph0, phk):
#        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
#        self.b += torch.sum((v0 - vk), 0)
#        self.a += torch.sum((ph0 - phk), 0)
#

#
## Test the RBM
#test_loss = 0
#s = 0.
#for id_user in range(nb_users):
#    v = training_set[id_user:id_user+1]
#    vt = test_set[id_user:id_user+1]
#    _, h = rbm.sample_h(v)
#    _, v = rbm.sample_v(h)
#    test_loss += torch.mean(torch.abs(vt - v))
#    s += 1.
#print('test loss: ' + str(test_loss/s))
#











