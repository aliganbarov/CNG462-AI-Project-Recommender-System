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


class PreprocessData:
    def __init__(self, train_file, test_file, c, conn):
        self.train_file = train_file
        self.test_file = test_file
        self._c = c
        self._conn = conn
        self._extra_files = []
        self._chunksize = 1000000
    
    def split_file (self):
        chunksize = self._chunksize - 1
        names = ["userid", "offerid", "countrycode", "category", "merchant", "utcdate", "rating"]
        user_id_counter = 0
        offer_id_counter = 0
        user_id_to_int_map = {}
        offer_id_to_int_map = {}
        nb_of_train_files = 0
        nb_of_test_files = 0
        file_name = "train/train"
        
        if not os.path.exists("train"):
            os.makedirs("train")
        if not os.path.exists("test"):
            os.makedirs("test")
        
        # Split train file into smaller files (1M rows each)
        i = 1
        try:
            for training_chunk in pd.read_csv(self.train_file, chunksize=chunksize, names=names):
                print("Creating file: " + file_name + str(i) + ".csv")
                training_chunk.to_csv(file_name + str(i) + ".csv", index=False)
                self._extra_files.append(file_name + str(i) + ".csv")
                i += 1
                nb_of_train_files += 1
        except FileNotFoundError:
            print("File not found!")
            return False
           
        # Split test file into smaller files (1M rows each)
        file_name = "test/test"
        i = 1    
        try:
            for testing_chunk in pd.read_csv(self.test_file, chunksize=chunksize, names=names):
                print("Creating file: " + file_name + str(i) + ".csv")
                testing_chunk.to_csv(file_name + str(i) + ".csv", index=False)
                self._extra_files.append(file_name + str(i) + ".csv")
                i += 1
                nb_of_test_files += 1
        except FileNotFoundError:
            print("File not found!")
            return False
        
        print("It is recommended if you resave created files according to your OS csv type (open & save again)")
        cont = input("Press enter when done")
        
        # replace unique values of column with unique numbers for train set
        def replace_index(data, my_dict, k):
            for i in range(0, data.shape[0]):
                if data[i] in my_dict:
                    data[i] = my_dict[data[i]]
                else:
                    my_dict[data[i]] = k
                    data[i] = k
                    k += 1
            return data, k
        
        # replace ids of user and offer with unique numbers for all train set
        chunksize = self._chunksize
        for i in range(1, nb_of_train_files + 1):
            print("Compressing.. " + "train/train" + str(i) + "_replaced.csv")
            for training_chunk in pd.read_csv('train/train' + str(i) + '.csv', chunksize=chunksize, names=names, skiprows = [0,1]):
                training_chunk = np.array(training_chunk, dtype='str')
                training_chunk[:, 0], user_id_counter = replace_index(training_chunk[:, 0], user_id_to_int_map, user_id_counter)
                training_chunk[:, 1], offer_id_counter = replace_index(training_chunk[:, 1], offer_id_to_int_map, offer_id_counter)
                df = pd.DataFrame(training_chunk)
                df.to_csv("train/train" + str(i) + "_replaced.csv", index=False)
                self._extra_files.append("train/train" + str(i) + "_replaced.csv")
        
        # replace known unique values from train set with unique numbers for test set
        def replace_index_test(data, my_dict):
            for i in range(0, data.shape[0]):
                if data[i] in my_dict:
                    data[i] = my_dict[data[i]]
            return data
        
        # replace ids of user and offer with unique numbers for all train set
        for i in range(1, nb_of_test_files + 1):
            print("Compressing.. " + "test/test" + str(i) + "_replaced.csv")
            for test_chunk in pd.read_csv('test/test' + str(i) + '.csv', chunksize=chunksize, names=names, skiprows = [0,1]):
                test_chunk = np.array(test_chunk, dtype='str')
                test_chunk[:, 0] = replace_index_test(test_chunk[:, 0], user_id_to_int_map)
                test_chunk[:, 1] = replace_index_test(test_chunk[:, 1], offer_id_to_int_map)
                df = pd.DataFrame(test_chunk)
                df.to_csv("test/test" + str(i) + "_replaced.csv", index=False)
                self._extra_files.append("test/test" + str(i) + "_replaced.csv")
        
    def remove_extra_files(self):
        for extra_file in self._extra_files:
            os.remove(extra_file)
    
    # writes data from files to database using SQLite3
    def write_to_database(self):
        chunksize = self._chunksize
        names = ["userid", "offerid", "countrycode", "category", "merchant", "utcdate", "rating"]
        nb_users = 291485
        nb_offers = 2158859
        # CONNECT TO DB
        c = self._c
        conn = self._conn
        # DROP ratings TABLE
        c.execute("DROP TABLE IF EXISTS ratings;")
        # CREATE RATINGS TABLE
        c.execute("CREATE TABLE ratings (user_id int,offer_id int, rating int);")

        
        print("Saving data in database..")
        # FILL IN RATINGS
        for i in range(1, 17):
            print("Inserting file: " + str(i))
            for training_chunk in pd.read_csv('train/train' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
                training_chunk = np.array(training_chunk, dtype='str')
                for i in range(2, len(training_chunk[:, 0])):
                    try:
                        c.execute("INSERT INTO ratings (user_id, offer_id, rating)  VALUES(?, ?, ?)", 
                              [int(training_chunk[:, 0][i]), int(training_chunk[:, 1][i]), int(training_chunk[:, 6][i])])
                    except ValueError:
                        pass
         
        # DROP ratings_test TABLE
        c.execute("DROP TABLE IF EXISTS ratings_test;")
        # CREATE ratings_test TABLE 
        c.execute("CREATE TABLE ratings_test(user_id int,offer_id int, rating int);")
        # FILL IN TEST RATINGS
        for i in range(1, 3):
            print("Inserting file: " + str(i))
            for training_chunk in pd.read_csv('test/test' + str(i) + '_replaced.csv', chunksize=chunksize, names=names):
                training_chunk = np.array(training_chunk, dtype='str')
                for i in range(2, len(training_chunk[:, 0])):
                    try:
                        c.execute("INSERT INTO ratings_test (user_id, offer_id, rating)  VALUES(?, ?, ?)", 
                              [int(training_chunk[:, 0][i]), int(training_chunk[:, 1][i]), int(training_chunk[:, 6][i])])
                    except ValueError:
                        pass
        conn.commit()
        c.close()  
