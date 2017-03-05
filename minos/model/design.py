'''
Created on Feb 14, 2017

@author: julien
'''
from copy import deepcopy
from random import Random, shuffle

from minos.experiment.experiment import Blueprint
from minos.experiment.training import Training
from minos.model.model import Layout, Row, Layer, Block,\
    Optimizer
from minos.model.parameter import random_initial_param_value,\
    random_list_element, mutate_param


rand = Random()


def create_random_blueprint(experiment):
    return Blueprint(
        _random_layout(
            experiment.layout,
            experiment.parameters),
        _random_training(experiment))


def _random_training(experiment):
    training = deepcopy(experiment.training)
    if experiment.parameters.is_optimizer_search():
        training.optimizer = _random_optimizer(
            training.optimizer,
            experiment.parameters)
    return training


def _random_optimizer(optimizer, experiment_parameters):
    ref_parameters = experiment_parameters.get_optimizers_parameters()
    optimizer_id = optimizer.optimizer if optimizer else None
    if not optimizer_id:
        optimizers = list(ref_parameters.keys())
        optimizer_id = optimizers[rand.randint(0, len(optimizers) - 1)]
    if not optimizer_id in ref_parameters:
        raise Exception('Unknown optimizer id: "%s"' % optimizer_id)
    param_space = deepcopy(ref_parameters[optimizer_id])
    if optimizer:
        param_space.update(optimizer.parameters)
    parameters = {
        name: random_initial_param_value(param)
        for name, param in param_space.items()}
    return Optimizer(optimizer_id, parameters)


def _random_layout(layout, experiment_parameters):
    layout = Layout(
        layout.input_size,
        layout.output_size,
        layout.output_activation,
        layout.block)
    if experiment_parameters.is_layout_search():
        rows = random_initial_param_value(experiment_parameters.get_layout_parameter('rows'))
        for row_idx in range(rows):
            layout.rows.append(
                _random_layout_row(
                    layout,
                    row_idx,
                    experiment_parameters))
    else:
        layout.rows = [
            Row(blocks=[
                _instantiate_layout_block(
                    layout,
                    0,
                    experiment_parameters)])]
    if experiment_parameters.is_parameters_search():
        _set_layout_random_parameters(layout, experiment_parameters)
    return layout


def _set_layout_random_parameters(layout, experiment_parameters):
    for layer in layout.get_layers():
        _set_layer_random_parameters(layer, experiment_parameters)


def _set_layer_random_parameters(layer, experiment_parameters):
    param_space = deepcopy(experiment_parameters.get_layer_parameters(layer.layer_type))
    if param_space is None:
        raise Exception('No parameters defined for layer %s' % layer.layer_type)
    param_space.update(layer.parameters)
    layer.parameters = {
        name: random_initial_param_value(param)
        for name, param in param_space.items()}
    layer.apply_constraints()


def _random_layout_row(layout, row_idx,
                       experiment_parameters, init_layer_parameters=False):
    blocks = random_initial_param_value(param=experiment_parameters.get_layout_parameter('blocks'))
    return Row([
        _random_layout_block(
            layout,
            row_idx,
            experiment_parameters,
            init_layer_parameters=init_layer_parameters)
        for _ in range(blocks)])


def _is_multiple_block_layouts_defined(layout):
    if not layout.block:
        return False
    if not all(isinstance(e, list) for e in layout.block):
        return False
    for element in layout.block:
        for layer in element:
            if not isinstance(layer, str)\
                    and not isinstance(layer, tuple):
                return False
    return True


def _instantiate_layout_block(layout, row_idx, experiment_parameters, init_layer_parameters=False):
    block_layout = layout.block
    if _is_multiple_block_layouts_defined(layout):
        block_layout = random_list_element(layout.block)
    block = Block()
    _setup_block_input(layout, row_idx, block, experiment_parameters)
    block.layers = _create_template_layers(
        block_layout,
        experiment_parameters,
        init_layer_parameters=init_layer_parameters)
    return block


def _random_layout_block(layout, row_idx,
                         experiment_parameters, init_layer_parameters=False):
    if layout.block:
        return _instantiate_layout_block(
            layout,
            row_idx,
            experiment_parameters,
            init_layer_parameters=init_layer_parameters)
    layers = random_initial_param_value(experiment_parameters.get_layout_parameter('layers'))
    template = []
    for _ in range(layers):
        allowed_layers = get_allowed_new_block_layers(template, experiment_parameters)
        if len(allowed_layers) == 0:
            break
        new_layer = random_list_element(allowed_layers)
        template.append(new_layer)
    block = Block()
    _setup_block_input(layout, row_idx, block, experiment_parameters)
    block.layers += _create_template_layers(
        template,
        experiment_parameters,
        init_layer_parameters)
    return block


def _setup_block_inputs(layout, experiment_parameters):
    for row_idx in range(len(layout.rows)):
        for block in layout.rows[row_idx].blocks:
            _setup_block_input(layout, row_idx, block, experiment_parameters)


def _setup_block_input(layout, row_idx, block, experiment_parameters):
    if row_idx == 0:
        if len(block.input_layers) > 0:
            block.input_layers = list()
        return
    previous_row = layout.get_rows()[row_idx - 1]
    if len(previous_row.get_blocks()) == 1:
        if len(block.input_layers) > 0:
            block.input_layers = list()
        return
    if len(block.input_layers) > 0:
        return
    template = layout.block_input
    if not template:
        template = [('Merge', dict(mode='concat'))]
    block.input_layers = _create_template_layers(template, experiment_parameters)
    for layer in block.input_layers:
        layer.apply_constraints()


def _get_layer_type(layer):
    layer_type = layer.layer_type\
        if isinstance(layer, Layer) else layer[0]
    return layer_type


def _is_first_layer_merge(layers):
    if len(layers) == 0:
        return False
    return _is_merge_layer(layers[0])


def _is_merge_layer(layer):
    layer_type = _get_layer_type(layer)
    return layer_type == 'Merge'


def _is_first_layers_match_template(layers, template):
    for idx in range(len(template)):
        if idx >= len(layers):
            return False
        layer_type = layers[idx].layer_type\
            if isinstance(layers[idx], Layer) else layers[idx][0]
        if template[idx][0] != layer_type:
            return False
    return True


def _create_template_layers(template, experiment_parameters, init_layer_parameters=False):
    layers = list()
    for layer_definition in template:
        layer_type = layer_definition[0]\
            if isinstance(layer_definition, tuple) else layer_definition
        parameter_constraints = deepcopy(layer_definition[1])\
            if isinstance(layer_definition, tuple) else dict()
        layer = Layer(
            layer_type,
            parameter_constraints=parameter_constraints)
        if init_layer_parameters:
            _set_layer_random_parameters(layer, experiment_parameters)
        layers.append(layer)
    return layers


def is_allowed_block_layer(layers, new_layer, parameters):
    previous_layer_type = None
    if len(layers) > 0:
        if isinstance(layers[-1], Layer):
            previous_layer_type = layers[-1].layer_type
        elif isinstance(layers[-1], tuple):
            previous_layer_type = layers[-1][0]
        else:
            previous_layer_type = layers[-1]
    if new_layer == 'BatchNormalization':
        return previous_layer_type == 'Dense'
    stackable_layers = parameters.get_layout_parameter('layer.stackable')
    is_stackable = stackable_layers and new_layer in stackable_layers.values
    return is_stackable or new_layer != previous_layer_type


def get_allowed_new_block_layers(layers, parameters):
    return [
        new_layer
        for new_layer in parameters.get_layer_types()
        if is_allowed_block_layer(layers, new_layer, parameters)]


def mutate_blueprint(blueprint, parameters,
                     p_mutate_layout=0.25, p_mutate_param=0.25,
                     layout_mutation_count=1, layout_mutables=None, mutate_in_place=True):
    if not mutate_in_place:
        blueprint = deepcopy(blueprint)
    if parameters.is_layout_search():
        if rand.random() < p_mutate_layout:
            _mutate_layout(
                blueprint.layout,
                parameters,
                mutables=layout_mutables,
                mutation_count=layout_mutation_count)
    if parameters.is_parameters_search():
        _mutate_layer_parameters(
            blueprint.layout,
            parameters,
            p_mutate_param=p_mutate_param)
    if parameters.is_optimizer_search():
        _mutate_optimizer(
            blueprint.training.optimizer,
            parameters,
            p_mutate_param=p_mutate_param)
    return blueprint


def _mutate_layout(layout, parameters,
                   mutables=None, mutation_count=1):
    mutables = mutables or ['rows', 'blocks', 'layers']
    shuffle(mutables)
    for mutation in mutables[:mutation_count]:
        if mutation == 'rows':
            _mutate_layout_rows(layout, parameters)
        if mutation == 'blocks':
            _mutate_layout_blocks(layout, parameters)
        if mutation == 'layers':
            _mutate_layout_layers(layout, parameters)
    _setup_block_inputs(layout, parameters)


def _mutate_layout_rows(layout, parameters):
    current_rows = len(layout.get_rows())
    new_rows = mutate_param(
        parameters.get_layout_parameter('rows'),
        current_rows)
    if new_rows > current_rows:
        for row_idx in range(current_rows, new_rows):
            layout.rows.append(
                _random_layout_row(
                    layout,
                    row_idx,
                    parameters,
                    init_layer_parameters=True))
    else:
        while len(layout.get_rows()) > new_rows:
            idx = rand.randint(0, len(layout.rows) - 1)
            layout.rows = [
                r
                for i, r in enumerate(layout.rows)
                if i != idx]


def _mutate_layout_blocks(layout, parameters):
    row = random_list_element(layout.rows)
    row_idx = layout.rows.index(row)
    current_blocks = len(row.get_blocks())
    new_blocks = mutate_param(
        parameters.get_layout_parameter('blocks'),
        current_blocks)
    if new_blocks > current_blocks:
        for _ in range(current_blocks, new_blocks):
            row.blocks.append(
                _random_layout_block(
                    layout,
                    row_idx,
                    parameters,
                    init_layer_parameters=True))
    else:
        while len(row.blocks) > new_blocks:
            idx = rand.randint(0, len(row.blocks) - 1)
            row.blocks = [
                b
                for i, b in enumerate(row.blocks)
                if i != idx]


def _mutate_layout_layers(layout, parameters):
    row = random_list_element(layout.rows)
    block = random_list_element(row.blocks)
    current_layers = len(block.get_layers())
    new_layers = mutate_param(
        parameters.get_layout_parameter('layers'),
        current_layers)
    if new_layers > current_layers:
        for _ in range(current_layers, new_layers):
            layers = get_allowed_new_block_layers(block.layers, parameters)
            block.layers += _create_template_layers(
                [random_list_element(layers)],
                parameters,
                init_layer_parameters=True)
    else:
        while len(block.layers) > new_layers:
            idx = rand.randint(0, len(block.layers) - 1)
            block.layers = [
                l
                for i, l in enumerate(block.layers)
                if i != idx]


def _mutate_layer_parameters(layout, parameters, p_mutate_param=0.1):
    for row in layout.rows:
        for block in row.blocks:
            for layer in block.layers:
                _mutate_layer(layer, parameters, p_mutate_param)


def _mutate_layer(layer, parameters, p_mutate_param=0.1):
    param_space = deepcopy(parameters.get_layer_parameters(layer.layer_type))
    for name, value in layer.parameters.items():
        if rand.random() < p_mutate_param:
            layer.parameters[name] = mutate_param(param_space[name], value)
    layer.apply_constraints()


def _mutate_optimizer(optimizer, parameters, p_mutate_param=0.1):
    param_space = deepcopy(parameters.get_optimizer_parameters(optimizer.optimizer))
    for name, value in optimizer.parameters.items():
        if rand.random() < p_mutate_param:
            optimizer.parameters[name] = mutate_param(param_space[name], value)


def mix_blueprints(blueprint1, blueprint2, parameters, p_mutate_param=0.05):
    parents = [blueprint1.layout, blueprint2.layout]
    layout = _mix_layouts(parents, parameters, p_mutate_param)
    parents = [blueprint1.training, blueprint2.training]
    training = _mix_trainings(parents, parameters, p_mutate_param)
    return Blueprint(layout, training)


def _mix_trainings(parents, parameters, p_mutate_param=0.05):
    training = Training(
        objective=parents[0].objective,
        optimizer=None,
        metric=parents[0].metric,
        stopping=parents[0].stopping,
        batch_size=parents[0].batch_size)
    parent_optimizers = [p.optimizer for p in parents]
    if parameters.is_optimizer_search():
        training.optimizer = _mix_optimizers(parent_optimizers)
        _mutate_optimizer(training.optimizer, parameters, p_mutate_param)
    else:
        training.optimizer = random_list_element(parent_optimizers)
    return training


def _mix_optimizers(parents):
    return deepcopy(random_list_element(parents))


def _mix_layouts(parent_layouts, parameters, p_mutate_param=0.05):
    layout = Layout(
        input_size=parent_layouts[0].input_size,
        output_size=parent_layouts[0].output_size,
        output_activation=parent_layouts[0].output_activation,
        block=parent_layouts[0].block,
        block_input=parent_layouts[0].block_input)
    if parameters.is_layout_search():
        rows = random_list_element([len(p.rows) for p in parent_layouts])
        for row_idx in range(rows):
            parent_rows = [p.rows[row_idx] for p in parent_layouts if row_idx < len(p.rows)]
            layout.rows.append(_mix_row(layout, row_idx, parent_rows, parameters))
        _setup_block_inputs(layout, parameters)
        if parameters.is_parameters_search():
            _mutate_layer_parameters(
                layout,
                parameters,
                p_mutate_param=p_mutate_param)
    else:
        layout.rows = random_list_element([
            parent_layouts[0].rows,
            parent_layouts[1].rows])
    return layout


def _mix_row(layout, row_idx, parent_rows, parameters):
    row = Row()
    blocks = random_list_element([len(p.blocks) for p in parent_rows])
    for block_idx in range(blocks):
        parent_blocks = [p.blocks[block_idx] for p in parent_rows if block_idx < len(p.blocks)]
        parent = random_list_element(parent_blocks)
        new_block = deepcopy(parent)
        row.blocks.append(new_block)
    return row
