# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:41 2018

@author: zl6415
"""
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
#from sklearn.neighbors import NearestNeighbor
from sklearn import preprocessing
import time
import sys
import json

# use 100 randomly selected identities from training set as validation set
# 1467 identities in total
num_identies = 1467
num_validation = 100  

camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
#only for testing the design
gallery_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
query_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)
features = np.asarray(features) 

features_train = features[train_idx-1,:]
# find distinct identities in training set (should be 767 refer to the protocol)
iden_train = np.unique(labels[train_idx-1,])
valid_label = np.random.choice(iden_train, num_validation)


# =============================================================================
# if __name__ == "__main__":
#     preprocessing()
# =============================================================================
