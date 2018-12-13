# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:42:41 2018

@author: zl6415
"""
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg # read images
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
from knn_naive import kNN
from ranklist import Rank
#from reidtool import visualize_ranked_results
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

def add_subplot_border(ax, width=1, color=None ):

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)


def plotimg(filename, ax):
    imgplot = mpimg.imread('PR_data/images_cuhk03/%s' %filename)
    ax.imshow(imgplot,aspect="auto")
    
# find distinct identities in training set (should be 767 refer to the protocol)
train_label = labels[train_idx-1]
iden_train = np.unique(labels[train_idx-1])
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
features_train = features[train_idx_new-1,:]
features_valid = features[valid_idx-1,:]
features_query = features[query_idx-1]
features_gallery = features[gallery_idx-1]
label_query = labels[query_idx-1]
label_gallery = labels[gallery_idx-1]
camId_query = camId[query_idx-1]
camId_gallery = camId[gallery_idx-1]
iden_query = np.unique(label_query)
iden_gallery = np.unique(label_gallery)

n_neighbors = 20
clf = kNN(n_neighbors,'euclidean')
pred_query, errors = clf.fit(features_query, features_gallery)
rk = Rank(n_neighbors)
arr_label_query = rk.generate(query_idx, pred_query, label_gallery,label_query, camId_query, camId_gallery)
#rank1 accuracy
score_test = accuracy_score(arr_label_query[:,0], label_query)



#ranlist virtualisation
plt.figure()
idx2plot = gallery_idx[pred_query][:,:10]
temp_idx = np.random.choice(1400,3)
idx2plot = idx2plot[temp_idx]
idx2plot_true = query_idx[temp_idx]
fig, axs = plt.subplots(nrows=3, ncols=11, figsize=(10, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
a = labels[idx2plot_true]
a1 = labels[idx2plot]
tf = a[:,None] == a1
label_query[temp_idx]
methods = list(range(33))
i = 0
tr = 0
ga = 0
fig.subplots_adjust(left=0.1, right=0.97, hspace=0.3, wspace=0.5)
for ax, interp_method in zip(axs.flat, methods):
    i+=1
    if (i == 1 or i == 12 or i == 23):
        name2plot = filelist[idx2plot_true[tr]]
        plotimg(name2plot[0],ax)
        add_subplot_border(ax,3,'k')
        tr+=1
#        print('!!!!%d' %i)
    else:
        l = int(ga % 100 / 10)
        s = ga%10 
#        print('l %d s %d ga %d' %(l,s,ga))
        name2plot = filelist[idx2plot[l][s]]
        plotimg(name2plot[0],ax)
        if tf[l][s] == True:
            col = 'g'
        if tf[l][s] == False:
            col = 'r'
        add_subplot_border(ax,3,col)
        ga+=1

plt.show()

