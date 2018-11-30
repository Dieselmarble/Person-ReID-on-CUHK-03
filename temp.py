# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:41 2018

@author: zl6415
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import time
import sys
import json

camId = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
labels = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
train_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()
#only for testing the design
gallery_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten()
query_idx = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
with open('PR_data/feature_data.json', 'r') as f:
    features = json.load(f)


#def main():



#if __name__ == "__main__":
#    main()