# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:41 2018

@author: zl6415
"""
import matplotlib
import matplotlib.image as mpimg # read images
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from knn_naive import kNN
from ranklist import Rank
import time
import sys
import json

# 1467 identities in total
num_identies = 1467
num_validation = 100  
rnd = np.random.RandomState(3)


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


def plotimg(filename):
    imgplot = mpimg.imread('PR_data/images_cuhk03/%s' %filename)
    #lena.shape #(512, 512, 3)
    plt.imshow(imgplot)
    
# find distinct identities in training set (should be 767 refer to the protocol)
train_label = labels[train_idx-1,]
iden_train = np.unique(labels[train_idx-1,])
#train_set = np.column_stack((train_idx,labels[train_idx-1,].T))

# use 100 randomly selected identities from training set as validation set
valid_iden = rnd.choice(iden_train, num_validation,replace=False)
valid_index = []
for i in range (num_validation):
    valid_index.append(np.argwhere(train_label == valid_iden[i]))

valid_index = np.concatenate(valid_index, axis=0)
valid_idx = train_idx[valid_index].ravel()
valid_label = labels[valid_idx-1]
train_idx_new = np.delete(train_idx, valid_index)
train_label_new = labels[train_idx_new-1]
features_train = features[train_idx_new-1]
features_valid = features[valid_idx-1]
features_query = features[query_idx-1]
features_gallery = features[gallery_idx-1]
label_query = labels[query_idx-1]
label_gallery = labels[gallery_idx-1]
camId_query = camId[query_idx-1]
camId_gallery = camId[gallery_idx-1]
iden_query = np.unique(label_query)
iden_gallery = np.unique(label_gallery)

print('start!')
lda = LinearDiscriminantAnalysis(n_components= 100)
#lda.fit(features_gallery,label_gallery)
#pred_labels = lda.predict(features_query)
#
#features_query2 = lda.transform(features_query)
#features_gallery2 = lda.transform(features_gallery)



for i in range (gallery_idx.shape[0]):
    if camId_gallery[i] == 1:
        features_gallery[i] = 0
features_gallery_ = features_gallery[~(features_gallery==0).all(1)]  
label_gallery_ = label_gallery[~(features_gallery==0).all(1)]  

for i in range (query_idx.shape[0]):
    if camId_query[i] == 2:
        features_query[i] = 0
features_query_ = features_query[~(features_query==0).all(1)]
label_query_ = label_query[~(features_query==0).all(1)]

lda.fit(features_gallery_,label_gallery_)
pred_labels = lda.predict(features_query_)


n_neighbors = 20
#knn classifier with metric defined
clf = kNN(n_neighbors,'euclidean')
rk = Rank(n_neighbors)

score_test = accuracy_score(pred_labels, label_query_)

#pred_query, errors = clf.fit(features_query2, features_gallery2)
#arr_label_query = rk.generate(query_idx, pred_query, label_gallery, label_query, camId_query, camId_gallery)
##rank1 test accuracy
#score_test = accuracy_score(arr_label_query[:,0], label_query)

#
#
#pca = decomposition.PCA(n_components=100)
#pca.fit(features_train)
#features_train_pca = pca.transform(features_train)
## setting up LMN
#lmnn = metric_learn.LMNN(k=2, learn_rate=1e-9,max_iter=2000, convergence_tol=0.01, 
#                         regularization = 0.5, use_pca= False, verbose=True)
#
## fit the data!
#lmnn.fit(features_train_pca, train_label_new)
#
#features_query_pca = pca.transform(features_query)
#features_gallery_pca = pca.transform(features_gallery)
#
## transform our input space
#X_lmnn = lmnn.transform()
#features_query2 = lmnn.transform(features_query_pca)
#features_gallery2 =lmnn.transform(features_gallery_pca)
#
#n_neighbors = 20
##knn classifier with metric defined
#clf = kNN(n_neighbors,'euclidean')
#pred, errors = clf.fit(features_query2, features_gallery2)

 
#pred_labels = label_gallery[pred]
#for i in range (query_idx.shape[0]):
#        if (pred_labels[i] == label_query[i]) and (camId_query[i] == camId_gallery[query_idx[i]):
#            pred_labels[i] = 0
 

 
#ranklist 
#arr_label = np.vstack(pred_labels_temp)
#
#rank1 accuracy
#score = accuracy_score(arr_label[:,0], label_query)


