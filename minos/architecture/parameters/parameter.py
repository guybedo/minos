'''
Created on Feb 8, 2017

@author: julien
'''
from inspect import isfunction, isclass


class Parameter(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value


class ParameterConstraint(object):

    levels = ['row', 'brick', 'block', 'layer']

    def __init__(self, namespace_id, name, value, **kwargs):
        self.namespace_id = namespace_id
        self.name = name
        self.value = value
        for level in self.levels:
            if level in kwargs:
                setattr(self, level, kwargs[level])

    def get_id(self):
        path = [
            level or '*'
            for level in self.levels]
        path.append(self.name)
        _id = to_absolute_id(self.namespace_id, self.name)
        return '%s:%s' % (
            '.'.join(path),
            _id)


def _to_str(element):
    if isinstance(element, str):
        return element
    if isclass(element) or isfunction(element):
        return element.__name__
    return str(element)


def to_namespace_id(path):
    if isinstance(path, list):
        return '.'.join([
            _to_str(_id)
            for _id in path])
    return path


def to_namespace_path(namespace_id):
    if isinstance(namespace_id, str):
        return namespace_id.split('.')
    if isinstance(namespace_id, list):
        return namespace_id
    raise Exception('Unsupported namespace_id format')


def to_absolute_id(namespace_id, name):
    return '%s/%s' % (
        to_namespace_id(namespace_id),
        name)


def boolean_param():
    return (True, False)


def float_param(lo=0., hi=1., default=0.):
    return (lo, hi, default)
