'''
Created on Feb 14, 2017

@author: julien
'''
from copy import deepcopy
from random import Random


from minos.experiment.experiment import Blueprint
from minos.experiment.training import Training
from minos.model.model import Layout, Row, Layer, Block,\
    Optimizer
from minos.model.parameter import random_param_value, str_param_name


rand = Random()


def create_random_blueprint(experiment):
    return Blueprint(
        _random_layout(
            experiment.layout,
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
    optimizer_id = optimizer.optimizer if optimizer else None
    if not optimizer_id:
        optimizers = list(ref_parameters.keys())
        optimizer_id = optimizers[rand.randint(0, len(optimizers) - 1)]
    param_space = deepcopy(ref_parameters[optimizer_id])
    if optimizer:
        param_space.update(optimizer.parameters)
    parameters = {
        name: random_param_value(param)
        for name, param in param_space.items()}
    return Optimizer(optimizer_id, parameters)


def _random_layout(layout, experiment_parameters):
    layout = Layout(
        layout.input_size,
        layout.output_size,
        layout.output_activation,
        layout.block)
    rows = random_param_value(experiment_parameters.get_layout_parameter('rows'))
    for _ in range(rows):
        layout.rows.append(
            _random_layout_row(
                layout,
                experiment_parameters))
    _apply_parameters_to_layout(layout, experiment_parameters)
    return layout


def _apply_parameters_to_layout(layout, experiment_parameters):
    for layer in layout.get_layers():
        _set_layer_random_parameters(layer, experiment_parameters)


def _set_layer_random_parameters(layer, experiment_parameters):
    param_space = deepcopy(experiment_parameters.get_layer_parameters(layer.layer_type))
    if not param_space:
        pass
    param_space.update(layer.parameters)
    layer.parameters = {
        name: random_param_value(param)
        for name, param in param_space.items()}


def _random_layout_row(layout, experiment_parameters):
    blocks = random_param_value(param=experiment_parameters.get_layout_parameter('blocks'))
    return Row([
        _random_layout_block(
            layout,
            experiment_parameters)
        for _ in range(blocks)])


def _random_layout_block(layout, experiment_parameters):
    template = layout.block
    if not template:
        layers = random_param_value(experiment_parameters.get_layout_parameter('layers'))
        template = []
        for _ in range(layers):
            block_layers = get_allowed_new_block_layers(template)
            if len(block_layers) == 0:
                break
            index = rand.randint(0, len(block_layers) - 1)
            template.append((block_layers[index], dict()))
    block = Block()
    _setup_block_input(layout, block, experiment_parameters)
    block.layers += _create_template_layers(template)
    return block


def _setup_block_input(layout, block, experiment_parameters):
    if len(layout.get_rows()) == 0:
        return
    template = layout.block_input
    if not template and len(layout.get_rows()[-1].get_blocks()) > 1:
        template = [('Merge', dict(mode='concat'))]
    if template:
        block.layers += _create_template_layers(template)


def _create_template_layers(template):
    return [
        Layer(
            str_param_name(layer[0] if isinstance(layer, tuple) else layer),
            deepcopy(layer[1]) if isinstance(layer, tuple) else dict())
        for layer in template]


def is_allowed_block_layer(layers, new_layer):
    if new_layer == 'BatchNormalization':
        return len(layers) > 0 and layers[-1][0] == 'Dense'
    return len(layers) == 0 or new_layer != layers[-1][0]


block_layers = ['Dense', 'Dropout', 'BatchNormalization']


def get_allowed_new_block_layers(layers):
    return [
        new_layer
        for new_layer in block_layers
        if is_allowed_block_layer(layers, new_layer)]


def mutate_blueprint(blueprint):
    pass


def mix_blueprints(blueprint1, blueprint2):
    pass
