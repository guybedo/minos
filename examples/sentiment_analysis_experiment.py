import logging
import os, sys

from examples.ga.dataset import get_reuters_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import boston_housing, imdb
import keras.metrics as metrics
import math

from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler

from minos.experiment.experiment import ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.model import Objective, Optimizer, Metric
from minos.model.parameter import int_param, float_param, Parameter

from minos.train.utils import SimpleBatchIterator, CpuEnvironment
from minos.train.utils import GpuEnvironment
from minos.utils import load_best_model

batch_size = 32
max_features = 10000
output_dim = 128
maxlen = 80  # cut texts after this number of words (among top max_features most common words)


def search_model(experiment_label, steps, batch_size):

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    #
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    batch_iterator = SimpleBatchIterator(X_train, y_train, batch_size=batch_size,autorestart=True)
    test_batch_iterator = SimpleBatchIterator(X_test, y_test, batch_size=batch_size,autorestart=True)

    from minos.experiment.experiment import Experiment

    from minos.model.model import Layout
    from minos.experiment.experiment import ExperimentParameters
    experiment_parameters = ExperimentParameters(use_default_values=True)
    experiment_settings = ExperimentSettings()

    training = Training(
        Objective('binary_crossentropy'),
        Optimizer(optimizer='Adam'),
        Metric('accuracy'),
        EpochStoppingCondition(10),
        batch_size)

    layout = Layout(
        maxlen,
        1,
        output_activation='sigmoid',
        block=[
            ('Embedding', { 'input_dim' : max_features , 'output_dim': 128}),
            # ('Embedding', {'input_dim': max_features }),
            # ('LSTM', { 'dropout': 0.2, 'recurrent_dropout': 0.2}),
            ('LSTM', { 'units': 128, 'dropout': 0.2, 'recurrent_dropout': 0.2}),
        ]

    )
    in_and_outs = Parameter(
            int,
            lo=100,
            hi=200,
            mutable=False)

    # experiment_parameters.layer_parameter('LSTM.units', in_and_outs)
    # experiment_parameters.layer_parameter('Embedding.output_dim', in_and_outs)


    experiment_parameters.layout_parameter('rows', 1)
    experiment_parameters.layout_parameter('blocks', 1)
    experiment_parameters.layout_parameter('layers', 1)


    experiment_settings.ga['population_size'] = 5
    experiment_settings.ga['generations'] = steps
    experiment_settings.ga['p_offspring'] = 1
    experiment_settings.ga['p_mutation'] = 1

    experiment = Experiment(
        experiment_label,
        layout=layout,
        training=training,
        batch_iterator=batch_iterator,
        test_batch_iterator=test_batch_iterator,
        # environment=GpuEnvironment(devices=['gpu:0'], n_jobs=10),
        environment=CpuEnvironment(),
        parameters=experiment_parameters,
        settings=experiment_settings
    )

    run_ga_search_experiment(
        experiment,
        resume=False,
        log_level='DEBUG')
    return load_best_model(experiment_label, steps - 1, X_train, y_train, X_test, y_test)


def main():
    label = 'sentiment_analysis'
    steps = 4

    # Load the model if it exists, otherwise do a new search
    # try:
    #     model = load_best_model(label, steps - 1, X_train, y_train, X_test, y_test, batch_size=1, epochs=10)
    # except Exception:
    model = search_model(label, steps,batch_size)



main()
