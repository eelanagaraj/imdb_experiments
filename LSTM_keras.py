from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl

from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from text_processing import load_imdb_data

'''
    Modified version of keras LSTM example: trains an LSTM network on 
    the imdb sentiment analysis data set. In addition to predicting on
    test data, also stores the model's weights and intermediate
    activation values for training and test data.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features=20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16

# had some luck with seed 111
print("Loading data...")
(X_train, y_train), (X_test, y_test), w = load_imdb_data(
    binary=True, max_features=max_features, maxlen=maxlen, seed=37)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True)
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, y_test)

print('Test accuracy:', acc)

store_weights = {}
for layer in model.layers :
    store_weights[layer] = layer.get_weights() 

# create a new model of same structure minus last layers, to explore intermediate outputs
print('Build truncated model')
chopped_model = Sequential()
chopped_model.add(Embedding(max_features, 256, weights=model.layers[0].get_weights()))
chopped_model.add(LSTM(256, 128, weights=model.layers[1].get_weights()))
chopped_model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

# pickle intermediate outputs, model weights
train_activations = chopped_model.predict(X_train, batch_size=batch_size)
test_activations = chopped_model.predict(X_test, batch_size=batch_size)
outputs = dict(final=classes, acc=acc, weights=store_weights, y_train=y_train, y_test=y_test,
    train_activations=train_activations, test_activations=test_activations) 

pkl.dump(outputs, open('results/predicted_activations.pkl', 'wb'), 
    protocol=pkl.HIGHEST_PROTOCOL)
