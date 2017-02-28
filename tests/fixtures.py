'''
Created on Feb 18, 2017

@author: julien
'''
import numpy

from keras import activations, backend
from keras.datasets import reuters
from keras.engine.topology import Layer
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from minos.train.utils import SimpleBatchIterator


def get_reuters_dataset(batch_size, max_words):
    (X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=max_words, test_split=0.2)

    tokenizer = Tokenizer(nb_words=max_words)
    X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
    X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')

    nb_classes = numpy.max(y_train) + 1
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


class CustomLayer(Layer):

    def __init__(self, output_dim=None, activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True)
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        activation = activations.get(self.activation)
        return activation(backend.dot(x, self.W) + self.b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': self.activation}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
