'''
Created on Feb 6, 2017

@author: julien
'''
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from minos.experiment.experiment import Experiment, ExperimentParameters,\
    load_experiment_blueprints
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, MetricDecreaseStoppingCondition,\
    EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.model import Objective, Optimizer, Metric, Layout
from minos.model.parameter import int_param, string_param, float_param
from minos.train.utils import CpuEnvironment, SimpleBatchIterator, cpu_device,\
    Environment

import numpy as np


np.random.seed(1337)
max_words = 1000


def get_dataset(batch_size):
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


def build_layout(input_size, output_size):
    return Layout(
        input_size=input_size,
        output_size=output_size,
        output_activation='softmax')


def custom_experiment_parameters():
    experiment_parameters = ExperimentParameters()
    experiment_parameters.layout_parameter('blocks', int_param(1, 5))
    experiment_parameters.layout_parameter('layers', int_param(1, 5))
    experiment_parameters.layer_parameter('Dense.output_dim', int_param(10, 500))
    experiment_parameters.layer_parameter('Dense.activation', string_param(['relu', 'tanh']))
    experiment_parameters.layer_parameter('Dropout.p', float_param(0.1, 0.9))
    return experiment_parameters


def metric_decrease_stopping_condition():
    return MetricDecreaseStoppingCondition(
        min_epoch=2,
        max_epoch=10,
        noprogress_count=5,
        measurement_interval=0.25)


def epoch_stopping_condition():
    return EpochStoppingCondition(epoch=10)


def search_model(experiment_label, steps, batch_size=32):
    batch_iterator, test_batch_iterator, nb_classes = get_dataset(batch_size)
    layout = build_layout(max_words, nb_classes)
    training = Training(
        Objective('categorical_crossentropy'),
        Optimizer(optimizer='Adam'),
        Metric('categorical_accuracy'),
        epoch_stopping_condition(),
        batch_size)
    parameters = custom_experiment_parameters()
    experiment = Experiment(
        experiment_label,
        layout,
        training,
        batch_iterator,
        test_batch_iterator,
        CpuEnvironment(n_jobs=2),
        parameters=parameters)
    run_ga_search_experiment(
        experiment,
        population_size=100,
        generations=steps,
        resume=False)


def load_best_model(experiment_label, step):
    blueprints = load_experiment_blueprints(
        experiment_label,
        step,
        Environment())
    return ModelBuilder().build(
        blueprints[0], 
        cpu_device(),
        compile_model=False)


def main():
    experiment_label = 'reuters_experiment'
    steps = 100
    search_model(experiment_label, steps)
    load_best_model(experiment_label, steps - 1)

if __name__ == '__main__':
    main()
