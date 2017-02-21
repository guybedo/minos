'''
Created on Feb 6, 2017

@author: julien
'''
from copy import deepcopy
import json
from os import path, makedirs
from os.path import join
import pickle

from minos.experiment.training import AccuracyDecreaseStoppingCondition
from minos.model.parameter import Parameter, str_param_name, expand_param_path
from minos.model.parameters import reference_parameters
from minos.train.utils import Environment
from minos.utils import setup_logging


class Experiment(object):

    def __init__(self, label, layout=None, training=None,
                 batch_iterator=None, test_batch_iterator=None,
                 environment=None, parameters=None, resume=False):
        self.label = label
        self.layout = layout
        self.training = training
        self.batch_iterator = batch_iterator
        self.test_batch_iterator = test_batch_iterator
        self.environment = environment
        self.parameters = parameters or ExperimentParameters()

    def get_experiment_data_dir(self):
        return join(
            self.environment.data_dir,
            self.label)

    def get_log_filename(self):
        return path.join(
            self.get_experiment_data_dir(),
            'experiment.log')

    def get_step_log_filename(self, step):
        return path.join(
            self.get_experiment_data_dir(),
            'experiment.step.%d.log' % step)

    def get_step_data_filename(self, step):
        return path.join(
            self.get_experiment_data_dir(),
            'experiment.step.%d.data' % step)

    def evaluate(self, blueprints):
        from minos.train.trainer import MultiProcessModelTrainer
        model_trainer = MultiProcessModelTrainer(
            self.batch_iterator,
            self.test_batch_iterator,
            self.environment)
        return [
            [result[1]]
            for result in model_trainer.build_and_train_models(blueprints)]


def setup_experiment(experiment, resume=False, log_level='INFO'):
    if not path.exists(experiment.get_experiment_data_dir()):
        makedirs(experiment.get_experiment_data_dir())
    setup_logging(
        experiment.get_log_filename(),
        log_level,
        resume=resume)


def experiment_step_logger(experiment, step, blueprints):
    generation_log_filename = experiment.get_step_log_filename(step)
    with open(generation_log_filename, 'w') as generation_file:
        individuals = [blueprint.todict() for blueprint in blueprints]
        json.dump(individuals, generation_file, indent=1, sort_keys=True)

    blueprints = [
        Blueprint(**{k: v for k, v in vars(blueprint).items() if k != 'fitness'})
        for blueprint in blueprints]
    generation_data_filename = experiment.get_step_data_filename(step)
    with open(generation_data_filename, 'wb') as generation_file:
        pickle.dump(blueprints, generation_file, -1)


def run_experiment(experiment, runner,
                   resume=False, log_level='INFO', **params):
    check_experiment_parameters(experiment)
    setup_experiment(experiment, resume, log_level)
    runner(
        experiment,
        step_logger=experiment_step_logger,
        **params)


class InvalidParametersException(Exception):

    def __init__(self, detail):
        super().__init__('Invalid parameters: %s' % detail)


def check_experiment_parameters(experiment):
    _assert_search_parameters_defined(experiment.parameters)
    if not experiment.parameters.is_layout_search()\
            and not experiment.layout.block:
        raise InvalidParametersException(
            'You have to specify a block template '
            + ' if you disable layout search')
    if experiment.parameters.is_layout_search()\
            and not experiment.parameters.is_parameters_search():
        raise InvalidParametersException(
            'If you do a layout search, '
            + ' you have to enable parameters search too')

    _assert_valid_training_parameters(experiment)


def _assert_valid_training_parameters(experiment):
    if not isinstance(
            experiment.training.stopping,
            AccuracyDecreaseStoppingCondition):
        return
    training = experiment.training
    if training.metric.metric != training.stopping.metric:
        raise InvalidParametersException(
            'The same metric must be used for training and early stopping, '
            + '%s != %s' % (
                training.metric.metric,
                training.stopping.metric))


def _assert_search_parameters_defined(experiment_parameters):
    for param_name in reference_parameters['search'].keys():
        value = experiment_parameters.get_search_parameter(param_name)
        if value is None or not isinstance(value, bool):
            raise InvalidParametersException('undefined search parameter: %s' % param_name)


def load_experiment_blueprints(experiment_label, step, environment=Environment()):
    experiment = Experiment(experiment_label, environment=environment)
    data_filename = experiment.get_step_data_filename(step)
    with open(data_filename, 'rb') as data_file:
        return pickle.load(data_file)


class ExperimentParameters(object):

    def __init__(self, use_default_values=True):
        self.parameters = deepcopy(reference_parameters)
        if use_default_values:
            self.parameters = self._init_default_values(self.parameters)

    def _init_default_values(self, node):
        if isinstance(node, Parameter):
            if node.default is not None or node.optional:
                return node.default
        else:
            for name, value in node.items():
                node[name] = self._init_default_values(value)
        return node

    def get_parameter(self, *path):
        node = self.parameters
        path = expand_param_path(path)
        for elem in path:
            if not elem in node:
                return None
            node = node[elem]
        return node

    def _set_parameter(self, _id, value):
        self.parameters = self._set_node_parameter(
            self.parameters,
            _id,
            value)

    def _set_node_parameter(self, node, path, value):
        node = node or dict()
        path = expand_param_path(path)
        if len(path) > 1:
            node[path[0]] = self._set_node_parameter(
                node[path[0]],
                path[1:],
                value)
        else:
            node[path[0]] = value
        return node

    def is_layout_search(self):
        return self.get_search_parameter('layout')

    def is_parameters_search(self):
        return self.get_search_parameter('parameters')

    def is_optimizer_search(self):
        return self.get_search_parameter('optimizer')

    def get_search_parameter(self, name):
        return self.get_parameter('search', name)

    def search_parameter(self, name, value):
        return self._set_parameter(
            ['search', name],
            value)

    def all_search_parameters(self, value):
        for param_name in self.get_parameter('search').keys():
            self.search_parameter(param_name, value)

    def layout_parameter(self, name, value):
        return self._set_parameter(
            ['layout', name],
            value)

    def get_layout_parameter(self, name):
        return self.get_parameter('layout', name)

    def layer_parameter(self, name, value):
        return self._set_parameter(
            ['layers', name],
            value)

    def get_layer_parameter(self, name):
        return self.get_parameter('layers', name)

    def get_layer_parameters(self, layer):
        return self.get_parameter('layers', str_param_name(layer))

    def optimizer_parameter(self, name, value):
        return self._set_parameter(
            ['optimizers', name],
            value)

    def get_optimizer_parameters(self, name):
        return self.get_parameter('optimizers', name)

    def get_optimizers_parameters(self):
        return self.get_parameter('optimizers')


class Blueprint(object):

    def __init__(self, layout, training):
        self.layout = layout
        self.training = training

    def todict(self):
        return {
            'layout': self.layout.todict(),
            'training': self.training.todict()}
