'''
Created on Feb 14, 2017

@author: julien
'''
from copy import deepcopy
from random import Random

from minos.experiment.experiment import Blueprint
from minos.experiment.training import Training
from minos.model.model import Layout, Row, Brick, block_layers, Layer, Block,\
    Optimizer
from minos.model.parameter import random_param_value, str_param_name


rand = Random()


def create_random_blueprint(experiment):
    return Blueprint(
        _random_layout(
            experiment.layout_definition,
            experiment.parameters),
        _random_training(experiment))


def _random_training(experiment):
    training = Training(**experiment.training.todict())
    training.optimizer = _random_optimizer(
        training.optimizer,
        experiment.parameters)
    return training


def _random_optimizer(optimizer, experiment_parameters):
    ref_parameters = experiment_parameters.get_optimizer_parameters()
    optimizer_id = optimizer.optimizer
    if not optimizer_id:
        optimizers = list(ref_parameters.keys())
        optimizer_id = optimizers[rand.randint(0, len(optimizers) - 1)]
    param_space = deepcopy(ref_parameters[optimizer_id])
    param_space.update(optimizer.parameters)
    parameters = {
        name: random_param_value(param)
        for name, param in param_space.items()}
    return Optimizer(optimizer_id, parameters)


def _random_layout(layout_definition, experiment_parameters):
    rows = random_param_value(experiment_parameters.get_layout_parameter('rows'))
    layout = Layout([
        _random_layout_row(layout_definition, experiment_parameters)
        for _ in range(rows)])
    _apply_parameters_to_layout(layout, experiment_parameters)
    return layout


def _apply_parameters_to_layout(layout, experiment_parameters):
    for layer in layout.get_layers():
        _set_layer_random_parameters(layer, experiment_parameters)


def _set_layer_random_parameters(layer, experiment_parameters):
    param_space = deepcopy(experiment_parameters.get_layer_parameters(layer.layer_type))
    param_space.update(layer.parameters)
    layer.parameters = {
        name: random_param_value(param)
        for name, param in param_space.items()}


def _random_layout_row(layout_definition, experiment_parameters):
    bricks = random_param_value(param=experiment_parameters.get_layout_parameter('bricks'))
    return Row([
        _random_layout_brick(layout_definition, experiment_parameters)
        for _ in range(bricks)])


def _random_layout_brick(layout_definition, experiment_parameters):
    blocks = random_param_value(experiment_parameters.get_layout_parameter('blocks'))
    return Brick([
        _random_layout_block(layout_definition, experiment_parameters)
        for _ in range(blocks)])


def _random_layout_block(layout_definition, experiment_parameters):
    if layout_definition.block_template:
        template = layout_definition.block_template
    else:
        index = rand.randint(0, len(block_layers) - 1)
        template = [(block_layers[index], dict())]
    layers = [
        Layer(
            str_param_name(layer[0] if isinstance(layer, tuple) else layer),
            deepcopy(layer[1]) if isinstance(layer, tuple) else dict())
        for layer in template]
    return Block(layers)


def mutate_blueprint(blueprint):
    pass


def mix_blueprints(blueprint1, blueprint2):
    pass
