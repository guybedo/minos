import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))
#
from sisy import run_sisy_experiment

layout = [('Input', {'units': 20, 'activation': 'relu'}),
          ('Dense', {'units': 64, 'activation': 'relu'}),
          ('Dense', {'units': 64, 'activation': 'relu'}),
          ('Output', {'units': 1, 'activation': 'sigmoid'}),
          ]
run_sisy_experiment(layout, 'sisy_binary_classifier_v1', (x_train, y_train), (x_test, y_test), epochs=40, batch_size=64,
                    loss='binary_crossentropy', optimizer='rmsprop', metric='accuracy')
