import math
import numpy as np
from sklearn import preprocessing

class PreferenceDataReader:
    def __init__(self):
        self.data = None
        self.target = None

    def create_train_test_split(self, split_fraction=0.7):
        size_test_set = math.floor((1 - split_fraction)*self.data.shape[0])
        
        self.train_data = self.data[:-size_test_set]
        self.test_data = self.data[-size_test_set:]
        self.train_target = self.target[:-size_test_set]
        self.test_target = self.target[-size_test_set:]

class FlixsterDataReader(PreferenceDataReader):
    def __init__(self, filepath, model_name):
        data_list, rating_list = self.read_input(filepath)
        raw_data = np.asarray(data_list)
        if model_name == "user_average":
            raw_data = raw_data[:, 0:1]
        elif model_name == "item_average":
            raw_data = raw_data[:, 1:2]
       
        self.enc = preprocessing.OneHotEncoder()
        self.enc.fit(raw_data)
        self.data = self.enc.transform(raw_data)
        self.target = np.asarray(rating_list)
    
    def read_input(self, filepath):
        f = open(filepath)
        data_list = []
        rating_list = []
        for line in f:
            cols = line.strip('\n').split("\t")
            user_id = int(cols[0])
            item_id = int(cols[1])
            rating = float(cols[2])
            data_list.append((user_id, item_id))
            rating_list.append(rating)
        return data_list, rating_list
            


