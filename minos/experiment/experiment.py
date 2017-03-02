'''
Created on Feb 6, 2017

@author: julien
'''
from copy import deepcopy
import json
import logging
from os import path, makedirs
from os.path import join
import pickle

from minos.experiment.training import AccuracyDecreaseStoppingCondition
from minos.model.parameter import Parameter, str_param_name, expand_param_path
from minos.model.parameters import reference_parameters, get_custom_layers,\
    get_custom_activations
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
        self.environment = environment or Environment()
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

    def get_checkpoint_filename(self):
        return path.join(
            self.get_experiment_data_dir(),
            'experiment.chkpt')

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


def experiment_step_logger(experiment, step, individuals):
    blueprints = [
        individual.to_blueprint()
        for individual in individuals]
    with open(experiment.get_step_log_filename(step), 'w') as generation_file:
        json.dump(
            [blueprint.todict() for blueprint in blueprints],
            generation_file,
            indent=1,
            sort_keys=True)
    with open(experiment.get_step_data_filename(step), 'wb') as generation_file:
        pickle.dump(blueprints, generation_file, -1)

    with open(experiment.get_checkpoint_filename(), 'wb') as checkpoint:
        data = {
            'step': step,
            'blueprints': blueprints}
        pickle.dump(data, checkpoint, -1)


def load_experiment_checkpoint(experiment):
    with open(experiment.get_checkpoint_filename(), 'rb') as checkpoint:
        data = pickle.load(checkpoint)
    return data['step'], data['blueprints']


def run_experiment(experiment, runner,
                   resume=False, log_level='INFO', **params):
    check_experiment_parameters(experiment)
    setup_experiment(experiment, resume, log_level)
    population = None
    population_age = 0
    if resume:
        logging.info('Trying to resume experiment')
        step, population = load_experiment_checkpoint(experiment)
        population_age = step + 1
    runner(
        experiment,
        step_logger=experiment_step_logger,
        population_age=population_age,
        population=population,
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


def load_experiment_best_blueprint(experiment_label, environment=Environment()):
    experiment = Experiment(experiment_label, environment=environment)
    last_step, _ = load_experiment_checkpoint(experiment)
    blueprints = list()
    for step in range(last_step):
        blueprint = load_experiment_step_best_blueprint(
            experiment_label,
            step,
            environment=environment)
        if blueprint:
            blueprints.append(blueprint)
    if len(blueprints) == 0:
        return None
    return list(sorted(blueprints, key=lambda b: -b.score[0]))[0]


def load_experiment_step_best_blueprint(experiment_label, step, environment=Environment()):
    blueprints = load_experiment_blueprints(
        experiment_label,
        step,
        environment)
    if len(blueprints) == 0:
        return None
    return list(sorted(blueprints, key=lambda b: -b.score[0]))[0]


def load_experiment_blueprints(experiment_label, step, environment=Environment()):
    experiment = Experiment(experiment_label, environment=environment)
    data_filename = experiment.get_step_data_filename(step)
    with open(data_filename, 'rb') as data_file:
        return pickle.load(data_file)


class ExperimentParameters(object):

    def __init__(self, use_default_values=True):
        self.parameters = deepcopy(reference_parameters)
        self._load_custom_definitions()
        if use_default_values:
            self.parameters = self._init_default_values(self.parameters)

    def _init_default_values(self, node):
        if isinstance(node, Parameter):
            if node.default is not None or node.optional:
                return node.default
        elif isinstance(node, dict):
            for name, value in node.items():
                node[name] = self._init_default_values(value)
        return node

    def _load_custom_definitions(self):
        for name, definition in get_custom_layers().items():
            self.layer_parameter(name, definition[1])
            self.add_layer_type(name, definition[2])

        for name in get_custom_activations().keys():
            self.add_activation(name)

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

    def add_activation(self, name):
        for _layer_name, parameters in self.get_parameter('layers').items():
            for param_name, param_value in parameters.items():
                if 'activation' == param_name and isinstance(param_value, Parameter):
                    param_value.values.append(name)

    def get_layer_types(self):
        param = self.get_layout_parameter('layer.type')
        if isinstance(param, list):
            return param
        elif isinstance(param, str):
            return [param]
        return self.get_layout_parameter('layer.type').values

    def layer_types(self, layer_types):
        self.layout_parameter('layer.type', layer_types)

    def add_layer_type(self, layer_type, stackable=False):
        layer_type_param = self.get_layout_parameter('layer.type')
        layer_type_param.values = list(set(layer_type_param.values + [layer_type]))
        if stackable:
            layer_stackable_param = self.get_layout_parameter('layer.stackable')
            layer_stackable_param.values = list(set(layer_stackable_param.values + [layer_type]))

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

    def __init__(self, layout, training, score=None):
        self.layout = layout
        self.training = training
        self.score = score

    def todict(self):
        return {
            'layout': self.layout.todict(),
            'training': self.training.todict(),
            'score': self.score}
