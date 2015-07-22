"""
helper functions to construct a vectorized representation of
text data, and to take a text vector and convert back to text 

keeping track of sources/code used:
	- keras helper code
	- http://ai.stanford.edu/~amaas/data/sentiment/ Stanford Maas, ACL 2011 paper
	- kaggle tutorial
"""
import cPickle as pkl
import numpy as np
import os
import nltk
import re
import random
import glob

from keras.preprocessing import sequence, text
from keras.utils import np_utils
from keras.datasets import imdb
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 

# define some global settings specific to imdb data set
NTRAIN = 25000
NTEST = 25000
M_LEN = 100
M_FEAT = 20000


""" take in a string, returns a string without stop words, html markup, etc"""
def clean_review (raw_text) :
    review_text = BeautifulSoup(raw_text, "html.parser").get_text()    
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english")) 
    # remove stop words                 
    meaningful_words = [w for w in words if not w in stops]   
    return str(" ".join( meaningful_words ))


""" process first num_files files in a directory and output list of strings
	and list of labels with corresponding indices """
def process_text_files(file_dir, num_files) :
	strings = [0]*num_files
	labels = [0]*num_files
	for filename in os.listdir(file_dir) :
		# index and rating contained in filename
		filepath = os.path.join(file_dir, filename)
		without_ext = filename.split('.')
		index, rating = without_ext[0].split('_')
		# only include files less than this num 
		index = int(index)
		rating = int(rating)
		if index < num_files :
			f = open(filepath, 'r')
			cleaned = clean_review(f.read())	
			strings[index] = cleaned
			labels[index] = rating
			f.close()
	return strings, labels


""" given a sequence matrix outputs a list of all words per sequence
    note: cannot preserve word ordering without restructuring a lot of
    keras' key functionality. """
def vector_to_word (sequence_matrix, word_dict) :
	wlists = []
	# invert key/val lookup in dictionary so numbers can be looked up
	w_dict_flipped = {y:x for x,y in word_dict.iteritems()}
	for row in sequence_matrix:
		wlist = []
		# convert each sequence into a word list
		for num in row :
			if num :
				wlist.append(w_dict_flipped[num])
		wlists.append(wlist)
	return wlists


""" Function to convert raw data in text files into sequences and to return a 
    corresponding list of labels (binary or categorical depending on setting). """
def text_preprocess (path_train_pos, path_train_neg, path_test_pos, path_test_neg, 
	num_train, num_test, max_features, maxlen, binary=True) :
	# process training data
	pos_train, ptrain_labels = process_text_files(path_train_pos, num_train/2)
	negs_train, ntrain_labels = process_text_files(path_train_neg, num_train/2)
	train_text = pos_train + negs_train

	# process test data
	pos_test, ptest_labels = process_text_files(path_test_pos, num_test/2)
	negs_test, ntest_labels = process_text_files(path_test_neg, num_test/2)
	test_text = pos_test + negs_test

	# process labels as either numerical categories or binary positive/negative
	if binary :
		y_train = [0]*num_train
		y_test = [0]*num_test
		for i in xrange(num_train/2) :
				y_train[i] = 1
		for i in xrange(num_test/2) :
				y_test[i] = 1
	else :
		y_train = ptrain_labels + ntrain_labels
		y_test = ptest_labels + ntest_labels

	# fit dictionary on all data
	tokenizer = text.Tokenizer(nb_words=max_features, lower=True, split=" ")
	tokenizer.fit_on_texts(train_text + test_text)
	w = tokenizer.word_index

	# convert training and test data into padded sequence arrays
	train_seq_list = tokenizer.texts_to_sequences(train_text)
	X_train = sequence.pad_sequences(train_seq_list, maxlen=maxlen)
	test_seq_list = tokenizer.texts_to_sequences(test_text)
	X_test = sequence.pad_sequences(test_seq_list, maxlen=maxlen)

	return (X_train, y_train), (X_test, y_test), w


""" Simple wrapper function: Opens processed files or processes raw data if
	necessary for binary or categorical use. Seed value allows for reproducibly
	random shuffling of the data set.	Return format:
		(X_train,y_train),(X_test,y_test),w 		"""
def load_imdb_data (binary, num_train=NTRAIN, num_test=NTEST, 
	max_features=M_FEAT, maxlen=M_LEN, seed=113) :
	# if num_files, max_features, maxlen are defaults, use standard file and shuffle
	if (num_train == NTRAIN and num_test == NTEST and 
		max_features == M_FEAT and maxlen == M_LEN) :
		if binary :
			unshuffled_data = pkl.load(open('data_processing/processed_imdb.pkl', 'rb'))
		else :
			unshuffled_data = pkl.load(open('data_processing/processed_imdb_cats.pkl', 'rb'))
		X_train = unshuffled_data['X_train']
		y_train = unshuffled_data['y_train']
		X_test = unshuffled_data['X_test']
		y_test = unshuffled_data['y_test']
		w = unshuffled_data['w']
	else :
	# re-process for different features
		(X_train, y_train), (X_test, y_test), w = text_preprocess(
			path_train_pos='data_processing/aclImdb/train/pos', 
			path_train_neg='data_processing/aclImdb/train/neg', 
			path_test_pos='data_processing/aclImdb/test/pos',
			path_test_neg='data_processing/aclImdb/test/neg',
			num_train=num_train, num_test=num_test, 
			max_features=max_features, maxlen=maxlen, binary=binary)
	# shuffle data in place with provided seed
	np.random.seed(seed)
	np.random.shuffle(X_train)
	np.random.seed(seed)
	np.random.shuffle(y_train)
	np.random.seed(seed)
	np.random.shuffle(X_test)
	np.random.seed(seed)
	np.random.shuffle(y_test)

	return (X_train, y_train), (X_test, y_test), w
	