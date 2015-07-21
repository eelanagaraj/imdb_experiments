
""" run k-Nearest Neighbors classification on imdb dataset as a basis
	for comparison with the LSTM model results. """

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from sklearn import neighbors
from matplotlib.colors import ListedColormap
from keras.datasets import imdb
from keras.preprocessing import sequence
import operator
import time
import random

""" use scikit learn's built-in k-nearest neighbors functionality
	to evaluate training and test data """
def kNN_fit (k, X_train, y_train, X_test, y_test, weights='distance'): 
	# make sure labels are numpy arrays; assume Xs are numpy arrays already
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)
	clf = neighbors.KNeighborsClassifier(k, weights='distance')
	clf.fit(X_train, y_train)
	acc = clf.score(X_test, y_test)
	return acc


""" compute matrix of all euclidean distances between points in the two sets
	For efficienty, matrix shape will be (nb_test, nb_train) """
def euclidean_distance(X_train, X_test) :
	nb_train = X_train.shape[0]
	nb_test = X_test.shape[0]
	# compute X_train_ip, e.g. ||Xi||^2, and X_test_ip, e.g. ||Xj||^2
	Xi = np.diag(np.dot(X_train,X_train.T))
	Xi = np.reshape(Xi,(nb_train,1))
	Xj = np.diag(np.dot(X_test,X_test.T))
	Xj = np.reshape(Xj,(1,nb_test))
	# compute 2<xi,xj>
	prod = -2*np.dot(X_train,X_test.T)
	# compute ||Xi-Xj||^2
	dist_matrix = Xi + prod + Xj
	assert (dist_matrix.shape == (nb_train,nb_test))
	return dist_matrix.T


""" return the 'majority vote' of the k-nearest neighbors. Arbitrary tiebreak. 
	Note: opted for a slightly slower implementation since it is
	cleaner than the alternative and works with binary or categorical. """
def majority_vote(y_train, neighbor_indices, binary=True) :
	votes = {}
	for i in neighbor_indices :
		# add votes for each category label in dictionary
		if y_train[i] in votes :
			votes[(y_train[i])] += 1
		else :
			votes[(y_train[i])] = 1	
	# return label key with most votes
	return (max(votes.iteritems(), key=operator.itemgetter(1))[0])


""" find the minimum k values per row (each row is a test set instance) 
	and the majority vote among those values. Calculate prediction error. """
def kNN_quick(k, X_train, y_train, X_test, y_test, binary=True) :
	nb_test = X_test.shape[0]
	predictions = [0]*(nb_test)
	k_neighs = [0]*k
	# find and sort distances
	dists = euclidean_distance(X_train, X_test)
	sorted_indices = np.argsort(dists)
	# each row: find indices of k closest points and majority vote
	for j in xrange(nb_test) :
		for x in xrange(k) :
			k_neighs[x] = sorted_indices[j][x]
		predictions[j] = majority_vote(y_train, k_neighs, binary)
	# evaluate predictions
	correct = 0
	for x in xrange(nb_test) :
		if (predictions[x] == y_test[x]) :
			correct += 1
	acc = correct / float(nb_test)
	return acc

# load all test data
ks = [15, 25, 50, 100]
outputs_binary = pkl.load(open('results/predicted_activations.pkl', 'rb'))
original_data = pkl.load(open('data_processing/processed_imdb.pkl', 'rb'))
X_train = outputs_binary['train_activations']
X_test = outputs_binary['test_activations']
y_train = original_data['y_train']
y_test = original_data['y_test']


# compare sklearn and my implementations of kNN
for k in ks :
	print 'k = ', k
	acc_sk = kNN_fit(k, X_train, y_train, X_test, y_test)
	acc_mine = kNN_quick(k, X_train, y_train, X_test, y_test)
	print 'scikit learn result = ', acc_sk
	print 'hand-coded result = ', acc_mine