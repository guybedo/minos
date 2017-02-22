'''
Created on Feb 8, 2017

@author: julien
'''


class Layout(object):

    def __init__(self, input_size, output_size,
                 output_activation, block=None, block_input=None, rows=None):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.output_activation = output_activation
        self.block = block
        self.block_input = block_input
        self.rows = rows or list()

    def get_rows(self):
        return self.rows

    def get_blocks(self):
        return [
            block
            for row in self.rows
            for block in row.get_blocks()]

    def get_layers(self):
        return [
            layer
            for row in self.rows
            for layer in row.get_layers()]

    def todict(self):
        dict_repr = dict(vars(self))
        del dict_repr['block']
        dict_repr['rows'] = [row.todict() for row in self.rows]
        return dict_repr


class Row(object):

    def __init__(self, blocks=None):
        self.blocks = blocks or list()

    def get_blocks(self):
        return self.blocks

    def get_layers(self):
        return [
            layer
            for block in self.blocks
            for layer in block.get_layers()]

    def todict(self):
        return {
            'blocks': [
                block.todict()
                for block in self.blocks]}


class Block(object):

    def __init__(self, layers=None, input_layers=None):
        self.layers = layers or list()
        self.input_layers = input_layers or list()

    def get_layers(self):
        return self.layers

    def todict(self):
        return {
            'input_layers': [
                layer.todict()
                for layer in self.input_layers],
            'layers': [
                layer.todict()
                for layer in self.layers]}


class Layer(object):

    def __init__(self, layer_type,
                 parameters=None, parameter_constraints=None):
        self.layer_type = layer_type
        self.parameters = parameters or dict()
        self.parameter_constraints = parameter_constraints or dict()

    def apply_constraints(self):
        self.parameters.update(self.parameter_constraints)

    def todict(self):
        return dict(vars(self))


class Metric(object):

    def __init__(self, metric):
        self.metric = metric

    def todict(self):
        return dict(vars(self))


class Objective(object):

    def __init__(self, objective):
        self.objective = objective

    def todict(self):
        return dict(vars(self))


class Optimizer(object):

    def __init__(self, optimizer=None, parameters=None):
        self.optimizer = optimizer
        self.parameters = parameters or dict()

    def todict(self):
        return dict(vars(self))
