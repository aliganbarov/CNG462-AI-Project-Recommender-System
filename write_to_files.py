import pandas as pd
import numpy as np

chunksize = 1111111
names = ["userid", "offerid", "countrycode", "category", "merchant", "utcdate", "rating"]

#file_name = "train/train"
#i = 1
#for training_chunk in pd.read_csv('train_de.csv', chunksize=chunksize, names=names):
#    training_chunk.to_csv(file_name + str(i) + ".csv", sep=',', encoding='utf-8')
#    i += 1
#   
    
#file_name = "test/test"
#i = 1    
#for testing_chunk in pd.read_csv('test_de.csv', chunksize=chunksize, names=names):
#    testing_chunk.to_csv(file_name + str(i) + ".csv", sep=',', encoding='utf-8')
#    i += 1

user_id_counter = 0
offer_id_counter = 0

user_id_to_int_map = {}
offer_id_to_int_map = {}
    
def replace_index(data, my_dict, k):
    for i in range(0, data.shape[0]):
        if data[i] in my_dict:
            data[i] = my_dict[data[i]]
        else:
            my_dict[data[i]] = k
            data[i] = k
            k += 1
    return data, k

#for i in range(1, 17):
#    for training_chunk in pd.read_csv('train/train' + str(i) + '.csv', chunksize=chunksize, names=names):
#        training_chunk = np.array(training_chunk, dtype='str')
#        training_chunk[:, 0], user_id_counter = replace_index(training_chunk[:, 0], user_id_to_int_map, user_id_counter)
#        training_chunk[:, 1], offer_id_counter = replace_index(training_chunk[:, 1], offer_id_to_int_map, offer_id_counter)
#        df = pd.DataFrame(training_chunk)
#        df.to_csv("train/train" + str(i) + "_replaced.csv")
        
def replace_index_test(data, my_dict):
    for i in range(0, data.shape[0]):
        if data[i] in my_dict:
            data[i] = my_dict[data[i]]
    return data

for i in range(1, 3):
    for test_chunk in pd.read_csv('test/test' + str(i) + '.csv', chunksize=chunksize, names=names):
        test_chunk = np.array(test_chunk, dtype='str')
        test_chunk[:, 0] = replace_index_test(test_chunk[:, 0], user_id_to_int_map)
        test_chunk[:, 1] = replace_index_test(test_chunk[:, 1], offer_id_to_int_map)
        df = pd.DataFrame(test_chunk)
        df.to_csv("test/test" + str(i) + "_replaced.csv")
    