from __future__ import print_function

from keras import Input
from keras.engine import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import *
from minos.model.model import Optimizer, Objective, Metric
from minos.model.parameter import int_param, float_param
from minos.train.utils import SimpleBatchIterator, GpuEnvironment
from minos.utils import load_best_model

batch_size = 32
max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

#inputs = Input(shape=(blueprint.layout.input_size,))
inputs = Input(shape=(maxlen,))
output_dim = 256
embedding = Embedding(input_dim=max_features, output_dim=output_dim)(inputs)
lstm = LSTM(output_dim)(embedding)
outputs = Dense(1, activation='sigmoid')(lstm)

model = Model(inputs=inputs, outputs=outputs)


# model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=batch_size, epochs=15, validation_data=(X_test, y_test))
#
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
#
print('Test score:', score)
print('Test accuracy:', acc)
#
