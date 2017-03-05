'''
Created on Feb 15, 2017

@author: julien
'''
import unittest

from minos.experiment.experiment import Experiment, ExperimentParameters,\
    check_experiment_parameters
from minos.experiment.training import Training
from minos.model.design import create_random_blueprint, mix_blueprints,\
    mutate_blueprint
from minos.model.model import Optimizer, Layout
from minos.model.parameter import is_valid_param_value, str_param_name,\
    string_param


class DesignTest(unittest.TestCase):

    def test_random_blueprint(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax')
        training = Training(
            objective=None,
            optimizer=None,
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None,
            parameters=ExperimentParameters(use_default_values=False))
        experiment.parameters.all_search_parameters(True)
        check_experiment_parameters(experiment)
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            self.assertIsNotNone(blueprint, 'Should have created a blueprint')

            optimizer = blueprint.training.optimizer
            self.assertIsNotNone(
                optimizer.optimizer,
                'Should have created an optimizer')
            ref_parameters = experiment.parameters.get_optimizers_parameters()
            for name, param in ref_parameters[optimizer.optimizer].items():
                self.assertTrue(
                    is_valid_param_value(param, optimizer.parameters[name]),
                    'Invalid param value')

            self.assertIsNotNone(blueprint.layout, 'Should have created a layout')
            rows = len(blueprint.layout.get_rows())
            self.assertTrue(
                is_valid_param_value(
                    experiment.parameters.get_layout_parameter('rows'),
                    rows),
                'Invalid value')
            for row in blueprint.layout.get_rows():
                blocks = len(row.get_blocks())
                self.assertTrue(
                    is_valid_param_value(
                        experiment.parameters.get_layout_parameter('blocks'),
                        blocks),
                    'Invalid value')
                for block in row.get_blocks():
                    layers = len([
                        l
                        for l in block.get_layers()
                        if l.layer_type != 'Merge'])
                    self.assertTrue(
                        is_valid_param_value(
                            experiment.parameters.get_layout_parameter('layers'),
                            layers),
                        'Invalid value')

    def test_predefined_parameters(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax')
        training = Training(
            objective=None,
            optimizer=Optimizer('SGD', {'lr': 1}),
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None)
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            self.assertIsNotNone(blueprint, 'Should have created a blueprint')

            self.assertEqual(
                training.optimizer.optimizer,
                blueprint.training.optimizer.optimizer,
                'Should have created an optimizer')
            self.assertEqual(
                blueprint.training.optimizer.parameters['lr'],
                training.optimizer.parameters['lr'],
                'Should have copied predefined parameter')

    def test_predefined_layer_type(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax')
        training = Training(
            objective=None,
            optimizer=Optimizer('SGD', {'lr': 1}),
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None)
        layer_types = ['Dropout']
        experiment.parameters.layer_types(string_param(layer_types))
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            self.assertIsNotNone(blueprint, 'Should have created a blueprint')
            for layer in blueprint.layout.get_layers():
                self.assertTrue(
                    layer.layer_type in layer_types,
                    'Should have used predefined layer types')

    def test_predefined_layout(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block=[
                ('Dense', {'activation': 'relu'}),
                'Dropout',
                ('Dense', {'output_dim': 100})])
        training = Training(
            objective=None,
            optimizer=Optimizer(),
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None,
            parameters=ExperimentParameters(use_default_values=False))
        experiment.parameters.search_parameter('layout', False)
        experiment.parameters.search_parameter('parameters', True)
        experiment.parameters.search_parameter('optimizer', True)
        for _ in range(10):
            blueprint1 = create_random_blueprint(experiment)
            blueprint2 = create_random_blueprint(experiment)
            blueprint3 = mix_blueprints(blueprint1, blueprint2, experiment.parameters)
            blueprint4 = mutate_blueprint(blueprint1, experiment.parameters, mutate_in_place=False)
            for idx, blueprint in enumerate([blueprint1, blueprint2, blueprint3, blueprint4]):
                self.assertIsNotNone(blueprint, 'Should have created a blueprint')
                self.assertIsNotNone(blueprint.layout, 'Should have created a layout')
                self.assertEqual(
                    1,
                    len(blueprint.layout.get_rows()),
                    'Should have 1 row')
                self.assertEqual(
                    1,
                    len(blueprint.layout.get_blocks()),
                    'Should have 1 block')
                self.assertEqual(
                    len(layout.block),
                    len(blueprint.layout.get_layers()),
                    'Should have predefined layers count')
                for i, row in enumerate(blueprint.layout.get_rows()):
                    blocks = len(row.get_blocks())
                    self.assertTrue(
                        is_valid_param_value(
                            experiment.parameters.get_layout_parameter('blocks'),
                            blocks),
                        'Invalid value')
                    for block in row.get_blocks():
                        self.assertEqual(
                            len(layout.block),
                            len(block.layers),
                            'Should have used template')
                        for i in range(len(layout.block)):
                            layer = layout.block[i]
                            layer_type = str_param_name(layer[0] if isinstance(layer, tuple) else layer)
                            params = layer[1] if isinstance(layer, tuple) else dict()
                            self.assertEqual(
                                layer_type,
                                block.layers[i].layer_type,
                                'Should have used the predefined layer type')
                            for name, value in params.items():
                                self.assertEqual(
                                    value,
                                    block.layers[i].parameters[name],
                                    'Should have used the predefined parameter value for blueprint %d' % idx)

    def test_predefined_multiple_blocklayout(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block=[
                [('Dense', {'activation': 'relu'})],
                [('Dense', {'activation': 'relu'}),
                    'Dropout',
                    ('Dense', {'output_dim': 100})]])
        training = Training(
            objective=None,
            optimizer=Optimizer(),
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None,
            parameters=ExperimentParameters(use_default_values=False))
        experiment.parameters.search_parameter('layout', False)
        experiment.parameters.search_parameter('parameters', True)
        experiment.parameters.search_parameter('optimizer', True)
        for _ in range(10):
            blueprint1 = create_random_blueprint(experiment)
            blueprint2 = create_random_blueprint(experiment)
            blueprint3 = mix_blueprints(blueprint1, blueprint2, experiment.parameters)
            blueprint4 = mutate_blueprint(blueprint1, experiment.parameters, mutate_in_place=False)
            for idx, blueprint in enumerate([blueprint1, blueprint2, blueprint3, blueprint4]):
                self.assertIsNotNone(blueprint, 'Should have created a blueprint')
                self.assertIsNotNone(blueprint.layout, 'Should have created a layout')
                self.assertEqual(
                    1,
                    len(blueprint.layout.get_rows()),
                    'Should have 1 row')
                self.assertEqual(
                    1,
                    len(blueprint.layout.get_blocks()),
                    'Should have 1 block')
                for i, row in enumerate(blueprint.layout.get_rows()):
                    blocks = len(row.get_blocks())
                    self.assertTrue(
                        is_valid_param_value(
                            experiment.parameters.get_layout_parameter('blocks'),
                            blocks),
                        'Invalid value')
                    for block in row.get_blocks():
                        self.assertTrue(
                            len(block.layers) == len(layout.block[0])
                            or len(block.layers) == len(layout.block[1]),
                            'Should have used template')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
