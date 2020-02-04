
# coding: utf-8

# # A KNN based classifier to classify given set of features in Mushroom Database

# In[16]:


import pandas as pd
from csv import reader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from math import sqrt
from numpy import array
from numpy import argmax
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report, accuracy_score, f1_score


# ## Definition of Data Attributes from Data_Set 

# ## The KNN-Algorithm

# In[17]:


class KNNClassifier:
    
    attributes= []
    dataset=[]
    test_data=[]
    new_dataset= pd.DataFrame()
    new_testset= pd.DataFrame()
    predicted_labels=[]
    num_neighbors= 3
    
    def design_attributes(self):
        cap_shape = ['b','c','x','f','k','s']
        cap_surface = ['f', 'g' , 'y', 's']
        cap_color = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']
        bruises = ['t', 'f']
        odor = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']
        gill_attach = ['a', 'f', 'd', 'n']
        gill_space = ['c', 'w' , 'd']
        gill_size = ['b', 'n']
        gill_color = ['k', 'n' , 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']
        stalk_shape = ['e', 't']
        stalk_sabove = ['f', 'y' , 'k', 's']
        stalk_sbelow = ['f', 'y' , 'k', 's']
        stalk_cabove = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
        stalk_cbelow = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
        veil_type = ['p', 'u']
        veil_color = ['n', 'o', 'w', 'y']
        ring_num = ['n', 'o', 't']
        ring_type = ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']
        spore = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']
        population = ['a', 'c', 'n', 's', 'v', 'y']
        habitat = ['g', 'l', 'm', 'p', 'u', 'w', 'd']


        self.attributes.append(cap_shape)
        self.attributes.append(cap_surface)
        self.attributes.append(cap_color)
        self.attributes.append(bruises)
        self.attributes.append(odor)
        self.attributes.append(gill_attach)
        self.attributes.append(gill_space)
        self.attributes.append(gill_size)
        self.attributes.append(gill_color)
        self.attributes.append(stalk_shape)
        self.attributes.append(stalk_sabove)
        self.attributes.append(stalk_sbelow)
        self.attributes.append(stalk_cabove)
        self.attributes.append(stalk_cbelow)
        self.attributes.append(veil_type)
        self.attributes.append(veil_color)
        self.attributes.append(ring_num)
        self.attributes.append(ring_type)
        self.attributes.append(spore)
        self.attributes.append(population)
        self.attributes.append(habitat)
    
    

    def train(self,filename):
        self.dataset= pd.read_csv(filename, header=None).to_numpy()
        self.dataset = np.delete(self.dataset, 11, 1)
#         return dataset

        self.design_attributes()
        s= []
        for i in range (0, len(self.dataset)):
            s.append(self.dataset[i][0])
        self.new_dataset= pd.DataFrame(s)

        self.one_hot_encode_dataset()
        self.new_dataset= self.new_dataset.to_numpy()
    
    def predict(self, filename):
#         print ("in predict")
        self.test_data= pd.read_csv(filename, header=None).to_numpy()
        self.test_data = np.delete(self.test_data, 10, 1)
        self.one_hot_encode_testdata()
        self.new_testset= self.new_testset.to_numpy()
        
        for row in self.new_testset:
            label = self.predict_classification(row)
            print (label)
            self.predicted_labels.append(label)
        return self.predicted_labels
        
    def one_hot_encode_dataset(self):
        for i in range (0, len(self.attributes)):
            s= []
            for j in range (0, len(self.dataset)):
                s.append(self.dataset[j][i+1])
            
            encode= pd.get_dummies(s, columns=self.attributes[i])
            encode = encode.T.reindex(self.attributes[i]).T.fillna(0)

            df= pd.DataFrame(encode)
            self.new_dataset= pd.concat([self.new_dataset, df], axis= 1)
        
#         print(self.new_dataset.head())
    
    def one_hot_encode_testdata(self):
        for i in range (0, len(self.attributes)):
            s= []
            for j in range (0, len(self.test_data)):
                s.append(self.test_data[j][i])
            
            encode= pd.get_dummies(s, columns=self.attributes[i])
            encode = encode.T.reindex(self.attributes[i]).T.fillna(0)

            df= pd.DataFrame(encode)
            self.new_testset= pd.concat([self.new_testset, df], axis= 1)
        
#         print(self.new_testset.head())
        
    def euclidean_distance(self,row1, row2):
        distance = 0.0
        for i in range(0,len(row1)):
            distance += (row1[i] - row2[i+1])**2
    #     print (sqrt(distance))
        return sqrt(distance)
    
    
    def get_neighbors(self, test_row):
        distances = list()
        for train_row in self.new_dataset:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][0])
    #         print (distances[i][0])
    #     print (neighbors)  
        return neighbors
    
    def predict_classification(self, test_row):
        neighbors = self.get_neighbors(test_row)
        output_values = [row[0] for row in neighbors]
    #     print(output_values)
        prediction = max(set(output_values), key=output_values.count)
        return prediction


# In[18]:


knn_classifier = KNNClassifier()
knn_classifier.train('/home/smriti/Downloads/train_2.csv')
predictions = knn_classifier.predict('/home/smriti/Downloads/test_2.csv')
test_labels = list()
with open("/home/smriti/Downloads/test_labels2.csv") as f:
    for line in f:
        test_labels.append(line.strip())

# print("Actual:" , test_labels, "Predicted:", predictions)    
# print (test_labels)
print (accuracy_score(test_labels, predictions))

