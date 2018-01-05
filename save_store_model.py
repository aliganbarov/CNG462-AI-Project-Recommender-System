# import libraries
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
import os

from preprocess_data import PreprocessData
from RBM import RBM


class Main:
    def __init__(self):
        nv = 2158860        # total # of offers 
        nh = 10
        self._rbm = RBM(nv, nh)
        self._conn = sqlite3.connect('example.db')
        self._c = self._conn.cursor()

    def menu(self):
        rbm = self._rbm
        c = self._c
        nb_users = 291485
        print("1) Preprocess data (DO ONLY ONCE)")
        print("2) Load stored model")
        print("3) Continue training current model")
        print("4) Train new model")
        print("5) Test data on curently loaded model")
        print("6) Exit")
        option = input("Choice: ")
        
        if option == '1':
            train_file_path = input("Enter train file path: ")
            test_file_path = input("Enter test file path: ")
            preprocessData = PreprocessData(train_file_path, test_file_path, c, self._conn)
            preprocessData.split_file()
            preprocessData.write_to_database()
            remove_files = input("Do you want to remove extra files? [y/n]: ")
            if remove_files == 'y' or remove_files == 'yes':
                preprocessData.remove_extra_files()
            
        elif option == '2':
            stored_model_name = "Enter stored model name (filename.bat): "
            try:
                rbm = torch.load(stored_model_name)
            except Exception:
                print("Failed to load file, please store model under folder 'models/'")

        elif option == '3':
            print("Coming soon...")
        
        elif option == '4':
            nh = input("Enter number of hidden nodes: ")
            nb_epoch = input("Enter number of epochs: ")
            batch_size = input("Enter batch size (You can train in small batches store model and continue training any other time): ")
            k = input("Enter k: ")
            try:
                nh = int(nh)
                nb_epoch = int(nb_epoch)
                batch_size = int(batch_size)
                k = int(k)
            except ValueError:
                print("Wrong parameters!")
                return
            start_training = input("Training can take long time.. Proceed? [y/n]: ")
            offer_id_counter = 2158860
            if start_training == 'y' or start_training == 'yes':
                training_set = []
                # TRAIN
                nb_epoch = 1
                start = timer()
                for epoch in range(1, nb_epoch + 1):
                    start_epoch = timer()
                    train_loss = 0
                    s = 0.
                    for current_user in range(0, nb_users):
                        if not current_user % batch_size:
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
                            for k in range(10):         # steps of random walk
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
                store_model = input("Do you want to store model? [y/n]: ")
                if store_model == 'y' or store_model == 'yes':
                    if not os.path.exists("train"):
                        os.makedirs("models")
                    file_name = input("Enter file name: ")
                    torch.save(rbm, file_name + ".bat")
            
        
        elif option == '5':
            # TEST
            test_set = []
            training_set = []
            test_loss = 0
            s = 0.
            start = timer()
            
            for current_user in range(0, nb_users):
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
        
        elif option == '6':
            exit()
        else:
            self.menu()
            
 
main = Main()
while True:
    main.menu()

