import logging

import sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import boston_housing
import keras.metrics as metrics
import math

from sklearn.preprocessing import StandardScaler

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.model import Objective, Optimizer, Metric
from minos.model.parameter import int_param, float_param

from minos.train.utils import SimpleBatchIterator, CpuEnvironment
from minos.train.utils import GpuEnvironment
from minos.utils import load_best_model

# Use the classic Boston Dataset to predict home prices

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

scale = StandardScaler()

# Scale our input data only , we're still trying to predict a continuous number for the house price
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)


def search_model(experiment_label, steps, batch_size=1):
    batch_iterator = SimpleBatchIterator(X_train, y_train, batch_size=1, autoloop=True)
    test_batch_iterator = SimpleBatchIterator(X_test, y_test, batch_size=1, autoloop=True)
    from minos.experiment.experiment import Experiment

    from minos.model.model import Layout

    layout = Layout(
        X_train.shape[1],  # Input size, 13 features I think
        1,  # Output size, we want just the price
        output_activation='linear',  # linear activation since its continous number
        output_initializer='normal',
        # Our template, just one block with two dense layers
        block=[
            ('Dense', {'kernel_initializer': 'normal', 'activation': 'relu'}),
            ('Dense', {'kernel_initializer': 'normal', 'activation': 'relu'})
        ]
    )

    # our training , MSE for the loss and metric, stopping condition of 5 since our epochs are only 10
    training = Training(
        Objective('mean_squared_error'),
        Optimizer(optimizer='Adam'),
        Metric('mean_squared_error'),
        EpochStoppingCondition(50),
        1)

    from minos.experiment.experiment import ExperimentParameters
    experiment_parameters = ExperimentParameters(use_default_values=True)
    experiment_settings = ExperimentSettings()

    experiment_parameters.layout_parameter('rows', 1)
    experiment_parameters.layout_parameter('blocks', 1)
    experiment_parameters.layout_parameter('layers', 1)
    experiment_parameters.layer_parameter('Dense.units', int_param(1, 20))

    experiment_settings.ga['population_size'] = 10
    experiment_settings.ga['generations'] = steps
    experiment_settings.ga['p_offspring'] = 1
    experiment_settings.ga['p_mutation'] = 1

    # TO specify minimizing the loss , lets use FitnessMin for a evolution criteria
    experiment_settings.ga['fitness_type'] = 'FitnessMin'

    experiment = Experiment(
        experiment_label,
        layout=layout,
        training=training,
        batch_iterator=batch_iterator,
        test_batch_iterator=test_batch_iterator,
        environment=GpuEnvironment(['gpu:0', 'gpu:1'], n_jobs=8),
        parameters=experiment_parameters,
        settings=experiment_settings
    )

    run_ga_search_experiment(
        experiment,
        resume=False,
        log_level='DEBUG')
    return load_best_model(experiment_label, steps - 1,X_train,y_train,X_test,y_test)


def main():
    label = 'regression_experiment_v22'
    steps = 5

    # Load the model if it exists, otherwise do a new search
    try:
        model = load_best_model(label, steps - 1, X_train, y_train, X_test, y_test, batch_size=1, epochs=50)
    except Exception:
        model = search_model(label, steps)

    yhat = model.predict(X_test, batch_size=1)
    for i in range(0, len(yhat)):
        yi = round(float(yhat[i][0]),2)
        yi_test = float(y_test[i])
        msg = "Predicted ${} - Actual = ${} - Difference = ({})".format(yi, yi_test, round(abs(yi_test - yi),2))
        logging.info(msg)
        print(msg)


main()
