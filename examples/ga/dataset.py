'''
Created on Feb 19, 2017

@author: julien
'''

from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from minos.train.utils import SimpleBatchIterator

import numpy as np


def get_reuters_dataset(batch_size, max_words):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)
    nb_classes = np.max(y_train) + 1
    tokenizer = Tokenizer(nb_words=max_words)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    batch_iterator = SimpleBatchIterator(
        X_train,
        y_train,
        batch_size,
        autoloop=True)
    test_batch_iterator = SimpleBatchIterator(
        X_test,
        y_test,
        len(X_test),
        autoloop=True)
    return batch_iterator, test_batch_iterator, nb_classes