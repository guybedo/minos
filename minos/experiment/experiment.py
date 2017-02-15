'''
Created on Feb 6, 2017

@author: julien
'''
from copy import deepcopy

from minos.model.parameter import Parameter, str_param_name
from minos.model.parameters import reference_parameters
from minos.train.trainer import MultiProcessModelTrainer


class Experiment(object):

    def __init__(self, label, layout_definition, training,
                 batch_iterator, test_batch_iterator, environment, resume=False):
        self.label = label
        self.layout_definition = layout_definition
        self.training = training
        self.batch_iterator = batch_iterator
        self.test_batch_iterator = test_batch_iterator
        self.environment = environment
        self.parameters = ExperimentParameters()

    def evaluate(self, blueprints):
        self._init_dataset()
        model_trainer = MultiProcessModelTrainer(
            self.training,
            self.batch_iterator,
            self.test_batch_iterator,
            self.environment)
        blueprints = [
            blueprint.todict(remove_property='fitness')
            for blueprint in blueprints]
        return [
            [result[1]]
            for result in model_trainer.build_and_train(blueprints)]


class ExperimentParameters(object):

    def __init__(self):
        self.parameters = deepcopy(reference_parameters)

    def get_parameter(self, *path):
        node = self.parameters
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
        if len(path) > 1:
            node[path[0]] = self._set_node_parameter(
                node[path[0]],
                path[1:],
                value)
        else:
            if not isinstance(value, Parameter):
                value = Parameter(path[0], value)
            node[path[0]] = value
        return node

    def layout_parameter(self, layout, name, value):
        return self._set_parameter(
            ['layout', layout, name],
            value)

    def get_layout_parameter(self, name):
        return self.get_parameter('layout', name)

    def layer_parameter(self, layer, name, value):
        return self._set_parameter(
            ['layers', str_param_name(layer), name],
            value)

    def get_layer_parameter(self, layer, name):
        return self.get_parameter('layers', str_param_name(layer), name)

    def get_layer_parameters(self, layer):
        return self.get_parameter('layers', str_param_name(layer))

    def optimizer_parameter(self, optimizer, name, value):
        return self._set_parameter(
            ['optimizers', str_param_name(optimizer), name],
            value)

    def get_optimizer_parameter(self, optimizer, name):
        return self.get_parameter('optimizers', str_param_name(optimizer), name)

    def get_optimizer_parameters(self):
        return self.get_parameter('optimizers')


class Blueprint(object):

    def __init__(self, layout, training):
        self.layout = layout
        self.training = training
