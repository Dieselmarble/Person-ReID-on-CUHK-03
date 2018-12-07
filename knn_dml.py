# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:29:55 2018

@author: zl6415
"""

from __future__ import absolute_import
import numpy as np
from sklearn import neighbors
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
import heapq

class kNN:
    """
        k-Nearest Neighbors (kNN)
        The nearest neighbors classifier adapted to be used with distance metric learning algorithms.

        Parameters
        ----------

        n_neighbors : int

            Number of neighbors to consider in classification.

        dml_algorithm : DML_Algorithm

            The distance metric learning algorithm that will provide the distance in kNN.
    """

    def __init__(self, n_neighbors, dml_algorithm):
        self.nn_ = n_neighbors
        self.dml = dml_algorithm
#        self.knn = neighbors.KNeighborsClassifier(n_neighbors)
#        self.knn_orig = neighbors.KNeighborsClassifier(n_neighbors)

    def distance(self,a,b,type_, iv):
        if type_ == 'euclidean':
            return distance.euclidean(a, b)
        elif type_ == 'manhattan':
            return distance.cityblock(a, b)
        elif type_ == 'chebyshev':
            return distance.chebyshev(a, b)
        elif type_ == 'cosine':
            return distance.cosine(a, b)
        elif type_ == 'correlation':
            return distance.correlation(a, b)
        elif type_ == 'mahalanobis':
            return distance.mahalanobis(a, b, iv)
        
    def fit(self,x_test,x_train, A):
        buffer_size = self.nn_
        pred2return = []
        dists2return = []
        for i in range(x_test.shape[0]): #1400
            dists = np.zeros(x_train.shape[0])
            for j in range(x_train.shape[0]): #5328
                dist = self.distance(x_test[i], x_train[j],self.dml, A)  
                dists[j] = dist
            #dists = heapq.nsmallest(buffer_size, dists)
            idx = dists.argsort()[:buffer_size] #indices of N minimum 
            dists_ = dists[idx]
            #pred = np.asarray(pred)
            dists2return.append(dists_)
            pred2return.append(idx)
        #return top N index in gallery and distances  
        return pred2return, dists2return

            
            
        
        