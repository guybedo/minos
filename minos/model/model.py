'''
Created on Feb 8, 2017

@author: julien
'''
from keras.layers.core import Dense, Dropout


class Layout(object):

    def __init__(self, rows):
        self.rows = rows

    def get_rows(self):
        return self.rows

    def get_bricks(self):
        return [
            brick
            for row in self.rows
            for brick in row.get_bricks()]

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


class Row(object):

    def __init__(self, bricks):
        self.bricks = bricks

    def get_bricks(self):
        return self.bricks

    def get_blocks(self):
        return [
            block
            for brick in self.bricks
            for block in brick.get_blocks()]

    def get_layers(self):
        return [
            layer
            for brick in self.bricks
            for layer in brick.get_layers()]


class Brick(object):

    def __init__(self, blocks):
        self.blocks = blocks

    def get_blocks(self):
        return self.blocks

    def get_layers(self):
        return [
            layer
            for block in self.blocks
            for layer in block.get_layers()]


class Block(object):

    def __init__(self, layers):
        self.layers = layers

    def get_layers(self):
        return self.layers


class Layer(object):

    def __init__(self, layer_type, parameters=None):
        self.layer_type = layer_type
        self.parameters = parameters or dict()


class LayoutDefinition(object):

    def __init__(self, input_size, output_size,
                 output_activation, block_template=None):
        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = output_activation
        self.block_template = block_template


class Metric(object):

    def __init__(self, metric):
        self.metric = metric


class Objective(object):

    def __init__(self, objective):
        self.objective = objective


class Optimizer(object):

    def __init__(self, optimizer=None, parameters=None):
        self.optimizer = optimizer
        self.parameters = parameters or dict()

block_layers = [Dense, Dropout]
