'''
Created on Feb 8, 2017

@author: julien
'''
from minos.architecture.parameters.constraint import constraints
from minos.architecture.parameters.layer import layers
from minos.architecture.parameters.optimizer import optimizers
from minos.architecture.parameters.parameter import Parameter, to_namespace_id, to_namespace_path,\
    to_absolute_id
from minos.architecture.parameters.regularizer import regularizers


class ParameterSpace(object):

    def __init__(self, **kwargs):
        self.parameters = dict()
        for domain, domain_parameters in kwargs.items():
            self._load_parameters(domain, domain_parameters)

    def _load_parameters(self, domain, domain_parameters):
        for namespace_id, parameters in domain_parameters.items():
            self._set_parameters(
                to_namespace_id([domain, namespace_id]),
                **parameters)

    def set_layer_parameters(self, layer, **kwargs):
        return self._set_parameters(
            to_namespace_id(['layers', layer]),
            **kwargs)

    def get_layer_parameter(self, layer, name):
        return self._get_parameter(
            to_namespace_id(['layers', layer]),
            name)

    def set_optimizer_parameters(self, optimizer, **kwargs):
        return self._set_parameters(
            to_namespace_id(['optimizers', optimizer]),
            **kwargs)

    def get_optimizer_parameter(self, optimizer, name):
        return self._get_parameter(
            to_namespace_id(['optimizers', optimizer]),
            name)

    def set_regularizer_parameters(self, regularizer, **kwargs):
        return self._set_parameters(
            to_namespace_id(['regularizers', regularizer]),
            **kwargs)

    def get_regularizer_parameter(self, regularizer, name):
        return self._get_parameter(
            to_namespace_id(['regularizers', regularizer]),
            name)

    def _set_parameters(self, namespace_id, **kwargs):
        for name, value in kwargs.items():
            self._set_parameter(namespace_id, name, value)
        return self

    def _get_parameter(self, namespace_id, name):
        return self.get_parameter(to_absolute_id(namespace_id, name))

    def get_parameter(self, absolute_id):
        absolute_id = absolute_id.split('/')
        namespace_id = absolute_id[0]
        _id = absolute_id[-1]
        return self._get_namespace(namespace_id).get(_id, None)

    def _set_parameter(self, namespace_id, name, value):
        if not isinstance(value, Parameter):
            value = Parameter(name, value)
        self._get_namespace(namespace_id)[name] = value

    def _get_namespace(self, namespace_id):
        namespace = self.parameters
        for elem in to_namespace_path(namespace_id):
            if not elem in namespace:
                namespace[elem] = dict()
            namespace = namespace[elem]
        return namespace


def build_parameter_space():
    return ParameterSpace(
        regularizers=regularizers,
        constraints=constraints,
        layers=layers,
        optimizers=optimizers)
