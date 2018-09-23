#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:14:48 2018

@author: poojithaamin
"""


#import the required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
from sklearn import metrics
import collections
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import preprocessing

from scipy import spatial
from math import*
import time
from sklearn.metrics.pairwise import cosine_similarity
import math as math
from numpy import dot
from numpy.linalg import norm

#text object to hold all the document features
text=[]


#Read the train data
file1 = open("/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment3/train.dat.txt")
lines = file1.readlines()

#Create a matrix of size 8581, 27673 with zero values
X = np.zeros((8581, 27673), dtype=int)

#Read the data from the train file and identify all the unique document features
for i in lines:
    j = i.split(' ')
    for k in range(len(j)):
        if k%2==0:
            if j[k] not in text:
                text.append(j[k])
count = 0  


#Populate the matrix X with term count values against the features    
for i in lines:
    j = i.split(' ')
    count = count+1
    for k in range(len(j)):
        if k%2==0:
            ind = text.index(j[k])
            X[count,ind] = j[k+1]

X = np.delete(X, (0), axis=0)

#Print values to varify the data    
print len(text)
print (X.shape)
print (text)
print (type(text))
X.shape
text.index('10448')

'''
DBSCAN clustering algorithm
Input: eps - distance, radius
MinSamples - Minimum number of points
'''

def getConnectedComponents(X, Pnt, eps, output):
    ConnectedComponents = []
    for Pnt2 in range(0, len(X)):
        #if (output[Pnt2] == 0) or (output[Pnt2] == -1):
            #numpy.sqrt(numpy.sum((x-y)**2))
            '''
            if spatial.distance.cosine(D[P], D[Pn]) < eps:
               neighbors.append(Pn)
            '''
            '''
            numerator = sum(a*b for a,b in zip(D[P], D[Pn]))
            denominator = square_root(D[P])*square_root(D[Pn])
            cosine= round(numerator/float(denominator),3)
            cosine = 1-cosine
            if cosine < eps:
               neighbors.append(Pn)
            '''
            cosine = 1-dot(X[Pnt], X[Pnt2])/(norm(X[Pnt])*norm(X[Pnt2]))
            if cosine < eps:
               ConnectedComponents.append(Pnt2)
       
    return ConnectedComponents


def dbscanImplementation(X, eps, MinSamples):   
    output = np.zeros((len(X)), dtype=int)
    clusterNo = 0
    visited = 0
    for Pnt in range(0, len(X)):
        #print Pnt
        if (output[Pnt] == 0):
            ConnectedComponents = getConnectedComponents(X, Pnt, eps, output)
            if len(ConnectedComponents) < MinSamples:
                output[Pnt] = -1
            else: 
               visited = 1
               clusterNo += 1
               connectClusters(X, output, Pnt, ConnectedComponents, clusterNo, eps, MinSamples)
    return output


def connectClusters(X, output, Pnt, ConnectedComponents, clusterNo, eps, MinSamples):
    output[Pnt] = clusterNo
    visited = 0
    idx = 0
    while idx < len(ConnectedComponents):    
    #for idx in range(0, len(ConnectedComponents)):
        print "hi"
        #output[idx] = clusterNo
        val = ConnectedComponents[idx]
        visited = 1
        #print visited
        if output[val] == -1:
           output[val] = clusterNo
        
        elif output[val] == 0:
            output[val] = clusterNo
            NewConnectedComponents = getConnectedComponents(X, val, eps, output)
            if MinSamples<=len(NewConnectedComponents):
                #ConnectedComponents = NewConnectedComponents
                ConnectedComponents = ConnectedComponents + NewConnectedComponents
        #print Pn, "output" ,np.count_nonzero(output==0)

        idx += 1             

def square_root(x):
    return round(sqrt(sum([a*a for a in x])),3)


'''
pca = decomposition.PCA(n_components=20, svd_solver='arpack', copy='False')
pca.fit(X)
X_New = pca.fit_transform(X)
X.shape
'''

#Get the term frequency-inverse document frequency
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
tfidfArray = tfidf.toarray()
tfidfArray.shape
tfidfArray[0]

#Perform dimensionality reduction
from sklearn.decomposition import PCA, KernelPCA
pca = KernelPCA(n_components=150, kernel='linear')
X_New = pca.fit_transform(tfidfArray)


#Standardise/Normalise the data
#X = StandardScaler().fit_transform(X)
X_normalized = preprocessing.normalize(X_New, norm='l2')


#X_New = StandardScaler().fit_transform(X_New)

#from sklearn.feature_selection import VarianceThreshold
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#X_New = sel.fit_transform(X)


#transformer = random_projection.GaussianRandomProjection()
#transformer = random_projection.SparseRandomProjection()
#X_New = transformer.fit_transform(X)

'''
Call the dbscan algorithm
Get the cluster values
Measure the time taken by the code
'''
start_time = time.time()
print ('Run dbscan')
my_labels = dbscanImplementation(X_normalized, eps=0.3, MinSamples=19)
elapsed_time = time.time() - start_time
set(my_labels)
collections.Counter(my_labels)  
print elapsed_time      

#Get the max value of the cluster
maxVal = max(my_labels)
     
#Assign noise points to cluster maxVal   
for i in range(0, len(my_labels)):
    if my_labels[i] == -1:
        my_labels[i] = maxVal 

#Write the cluster labels  to output file
my_labels.tofile('/Users/poojithaamin/Desktop/DOCS/SJSU/SEM3/255/Assignment3/output.dat.txt', sep="\n", format="%s")


#######################################################

#Code to get silhouette_score

'''
from sklearn import metrics
from sklearn.metrics import pairwise_distances
metrics.silhouette_score(X_New, skl_labels, metric='cosine')


from sklearn.metrics import silhouette_score

s = []
for mins in range(1,24):
    mins = mins+2
    my_labels = dbscanImplementation(X_normalized, eps=0.3, MinSamples=mins)
    labels = my_labels

    s.append(silhouette_score(X_normalized, labels, metric='cosine'))

plt.plot(s)
plt.ylabel("Silouette")
plt.xlabel("Mins")
plt.title("Silouette for DBSCAN Clusters Obtained")
'''