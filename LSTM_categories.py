from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import cPickle as pkl

from text_processing import load_imdb_data
from keras.preprocessing import sequence, text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU

'''
    Train an LSTM on rating classification -- non-binary sentiment classification.
    Try to guess the numerical rating that corresponds with the text review.

    This one doesn't do so well; haven't messed with configurations much and there
    are many that could likely be made to improve it from it's current <10%
    prediction rate. LSTM is maybe not so helpful with this categorization problem.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

max_features=20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
batch_size = 16

print("Loading data...")
(X_train, y_train), (X_test, y_test), w = load_imdb_data('data_processing/processed_imdb_cats.pkl', 
    binary=False, seed=113, maxlen=maxlen, max_features=max_features)

# for categories, convert label lists to binary arrays
nb_classes = np.max(y_train)+1
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256))
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, nb_classes, init='normal'))
model.add(Activation('softmax'))

# SO MANY LAYERS AHH WOW


# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, Y_test)

print('Test accuracy:', acc)

store_weights = {}
for layer in model.layers :
    store_weights[layer] = layer.get_weights() 

# create a new model of same structure minus last layers, to explore intermediate outputs
print('Build truncated model')
chopped_model = Sequential()
chopped_model.add(Embedding(max_features, 256))
chopped_model.add(LSTM(256, 128))
chopped_model.add(Dense(128, nb_classes))
chopped_model.set_weights(model.get_weights())
chopped_model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")

# pickle intermediate activations, model weights, accuracy
train_activations = chopped_model.predict(X_train, batch_size=batch_size)
test_activations = chopped_model.predict(X_test, batch_size=batch_size)
outputs = dict(final=classes, train_activations=train_activations, test_activations=test_activations, acc=acc) 
pkl.dump(outputs, open('results/predicted_activations_categories.pkl', 'wb'), 
    protocol=pkl.HIGHEST_PROTOCOL)