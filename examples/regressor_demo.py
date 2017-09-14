import logging
from pprint import pprint
import matplotlib.pyplot as plt

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout
    from keras.datasets import boston_housing
    import keras.metrics as metrics
    import math

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.model import Objective, Optimizer, Metric
from minos.model.parameter import int_param, float_param

from minos.train.utils import SimpleBatchIterator, CpuEnvironment
from minos.train.utils import GpuEnvironment
from minos.utils import load_best_model
import numpy as np

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


def base_model():
    model = Sequential()
    model.add(Dense(14, input_dim=13, init='normal', activation='relu'))
    model.add(Dense(7, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mse'])
    return model


seed = 7
np.random.seed(seed)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

clf = KerasRegressor(build_fn=base_model, nb_epoch=100, batch_size=1, verbose=2)

history = clf.fit(X_train, y_train)

res = clf.predict(X_test)
print(y_test)
print(res)



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
#plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()