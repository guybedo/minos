'''
Created on Feb 8, 2017

@author: julien
'''

from inspect import isfunction, isclass
from random import Random

import numpy


rand = Random()


class ParameterConstraint(object):

    levels = ['row', 'block', 'layer']

    def __init__(self, namespace_id, name, value, **kwargs):
        self.namespace_id = namespace_id
        self.name = name
        self.value = value
        for level in self.levels:
            if level in kwargs:
                setattr(self, level, kwargs[level])


class Parameter(object):

    def __init__(self, param_type, values=None,
                 lo=None, hi=None, default=None, optional=False, mutable=True):
        self.param_type = param_type
        self.values = values
        self.lo = lo
        self.hi = hi
        self.default = default
        self.optional = optional
        self.mutable = mutable


class Function(object):

    def __init__(self, func, params):
        self.func = func
        self.params = params


def str_param_name(element):
    if isinstance(element, str):
        return element
    if isclass(element) or isfunction(element):
        return element.__name__
    return str(element)


def param_id(path):
    return '.'.join([str_param_name(e) for e in path])


def param_path(_id):
    return _id.split('.')


def expand_param_path(path):
    path = [
        e
        for p in path
        for e in p.split('.')]
    return path


def boolean_param(default=None, optional=False):
    return Parameter(
        bool,
        (True, False),
        default=default,
        optional=optional)


def float_param(lo=0., hi=1., default=None, values=None, optional=False):
    return Parameter(
        float,
        lo=lo,
        hi=hi,
        values=values,
        default=default,
        optional=optional)


def int_param(lo=0, hi=100, default=None, values=None, optional=False):
    return Parameter(
        int,
        lo=lo,
        hi=hi,
        values=values,
        default=default,
        optional=optional)


def string_param(values, default=None, optional=False, mutable=True):
    return Parameter(
        str,
        values=values,
        default=default,
        optional=optional,
        mutable=mutable)


def param(values, default=None, optional=False):
    return Parameter(
        dict,
        values=values,
        default=default,
        optional=optional)


def func_param(function_definitions, default=None, optional=False):
    return Parameter(
        Function,
        values=function_definitions,
        default=default)


def is_valid_param_value(param, value):
    if not isinstance(value, param.param_type):
        return False
    if param.values is not None and len(param.values) > 0:
        return value in param.values
    if param.lo is not None and value < param.lo:
        return False
    if param.hi is not None and value > param.hi:
        return False
    return True


def mutate_param(param, value):
    try:
        if isinstance(param, dict):
            return {
                name: mutate_param(nested, value.get(name, None))
                for name, nested in param.items()}
        if not isinstance(param, Parameter):
            return value
        if param.optional and value is not None and rand.random() < 0.5:
            return None
        if param.values:
            if len(set(param.values)) == 1:
                return param.values[0]
            element = random_list_element(param.values)
            while element == value:
                element = random_list_element(param.values)
            return element
        elif param.lo == param.hi:
            return value
        value = value or 0
        std = (param.hi - param.lo) / 10
        if param.param_type == int:
            std = max(1, std)
        new_value = value + param.param_type(numpy.random.normal(0, std, 1)[0])
        while not is_valid_param_value(param, new_value) or new_value == value:
            new_value = value + param.param_type(numpy.random.normal(0, std, 1)[0])
        return new_value
    except Exception as ex:
        raise


def _is_optional_param(param):
    return isinstance(param, Parameter) and param.optional


def _random_dict_param_value(param):
    keys = [
        key
        for key in param.keys()
        if not _is_optional_param(param[key])
        or rand.random() < 0.5]
    return {
        key: random_param_value(param[key])
        for key in keys}


def _random_list_param_value(param):
    if len(param) == 0:
        return None
    if len(param) == 1:
        return param[0]
    if len(param) == 2:
        param = Parameter(type(param[0]), lo=param[0], hi=param[1])
    elif len(param) > 2:
        param = Parameter(type(param[0]), values=param)
    return random_param_value(param)


def random_param_value(param):
    if isinstance(param, dict):
        return _random_dict_param_value(param)
    if isinstance(param, list) or isinstance(param, tuple):
        return _random_list_param_value(param)
    if not isinstance(param, Parameter):
        return param
    value = 0
    if param.values:
        value = random_list_element(param.values)
    elif param.lo is not None and param.hi is not None:
        if param.param_type == int:
            value = rand.randint(param.lo, param.hi)
        else:
            value = param.lo + (rand.random() * (param.hi - param.lo))
    elif param.lo is not None:
        value = param.lo * (1 + rand.random())
    elif param.hi is not None:
        value = rand.random() * param.hi
    return param.param_type(value)


def random_initial_param_value(param):
    if isinstance(param, dict):
        return _random_dict_param_value(param)
    if isinstance(param, list) or isinstance(param, tuple):
        return _random_list_param_value(param)
    if not isinstance(param, Parameter):
        return param
    if param.values:
        return random_list_element(param.values)

    initial_value = None
    if param.default is not None:
        initial_value = param.default
    elif param.lo is not None:
        initial_value = param.lo
    else:
        initial_value = 0
    if rand.random() < 0.5:
        return mutate_param(param, initial_value)
    return initial_value


def random_list_element(elements):
    if len(elements) == 0:
        return None
    if len(elements) == 1:
        return elements[0]
    return elements[rand.randint(0, len(elements) - 1)]


def default_param_value(param):
    if isinstance(param, dict):
        return {
            name: default_param_value(value)
            for name, value in param.items()}
    elif isinstance(param, Parameter):
        if param.default is not None or param.optional:
            return param.default
    else:
        return param
