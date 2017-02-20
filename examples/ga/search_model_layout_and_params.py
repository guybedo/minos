'''
Created on Feb 6, 2017

@author: julien
'''
from minos.experiment.experiment import Experiment, ExperimentParameters,\
    load_experiment_blueprints
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, AccuracyDecreaseStoppingCondition,\
    EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.model import Objective, Optimizer, Metric, Layout
from minos.model.parameter import int_param, string_param, float_param
from minos.train.utils import CpuEnvironment, cpu_device, Environment

from examples.ga.dataset import get_reuters_dataset
import numpy as np


np.random.seed(1337)
max_words = 1000


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
    return AccuracyDecreaseStoppingCondition(
        min_epoch=2,
        max_epoch=10,
        noprogress_count=5,
        measurement_interval=0.25)


def epoch_stopping_condition():
    return EpochStoppingCondition(epoch=10)


def search_model(experiment_label, steps, batch_size=32):
    batch_iterator, test_batch_iterator, nb_classes = get_reuters_dataset(batch_size, max_words)
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
