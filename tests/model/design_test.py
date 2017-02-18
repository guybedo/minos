'''
Created on Feb 15, 2017

@author: julien
'''
import unittest

from keras.layers.core import Dense, Dropout

from minos.experiment.experiment import Experiment, ExperimentParameters
from minos.experiment.training import Training
from minos.model.design import create_random_blueprint, mutate_blueprint
from minos.model.model import Optimizer, Layout
from minos.model.parameter import is_valid_param_value, str_param_name


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
                    layers = len(block.get_layers())
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

    def test_predefined_layout(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block=[
                (Dense, {'activation': 'relu'}),
                Dropout,
                (Dense, {'output_dim': 100})])
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
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            self.assertIsNotNone(blueprint, 'Should have created a blueprint')

            self.assertIsNotNone(blueprint.layout, 'Should have created a layout')
            rows = len(blueprint.layout.get_rows())
            self.assertTrue(
                is_valid_param_value(
                    experiment.parameters.get_layout_parameter('rows'),
                    rows),
                'Invalid value')
            for i, row in enumerate(blueprint.layout.get_rows()):
                blocks = len(row.get_blocks())
                self.assertTrue(
                    is_valid_param_value(
                        experiment.parameters.get_layout_parameter('blocks'),
                        blocks),
                    'Invalid value')
                expected_layer_count = len(layout.block)
                input_layer_count = 0
                if i > 0 and len(blueprint.layout.get_rows()[i - 1].get_blocks()) > 1:
                    input_layer_count = 1
                    expected_layer_count += input_layer_count
                for block in row.get_blocks():
                    self.assertEqual(
                        expected_layer_count,
                        len(block.layers),
                        'Should have used template')
                    for i in range(input_layer_count):
                        self.assertEqual(
                            'Merge',
                            block.layers[i].layer_type,
                            'Should have used the predefined layer type')
                    for i in range(input_layer_count, len(layout.block)):
                        layer = layout.block[i]
                        layer_type = str_param_name(layer[0] if isinstance(layer, tuple) else layer)
                        params = layer[1] if isinstance(layer, tuple) else dict()
                        self.assertEqual(
                            layer_type,
                            block.layers[i + input_layer_count].layer_type,
                            'Should have used the predefined layer type')
                        for name, value in params.items():
                            self.assertEqual(
                                value,
                                block.layers[i + input_layer_count].parameters[name],
                                'Should have used the predefined parameter value')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
