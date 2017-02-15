'''
Created on Feb 15, 2017

@author: julien
'''
import unittest

from keras.layers.core import Dense, Dropout, Activation

from minos.experiment.experiment import Experiment
from minos.experiment.training import Training
from minos.model.design import create_random_blueprint
from minos.model.model import LayoutDefinition, Optimizer
from minos.model.parameter import is_valid_param_value, str_param_name


class DesignTest(unittest.TestCase):

    def test_random_blueprint(self):
        layout_definition = LayoutDefinition(
            input_size=100,
            output_size=10,
            output_activation='softmax')
        training = Training(
            objective=None,
            optimizer=Optimizer(),
            metric=None,
            stopping=None,
            batch_size=None)
        experiment = Experiment(
            'test',
            layout_definition,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None)
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            self.assertIsNotNone(blueprint, 'Should have created a blueprint')

            optimizer = blueprint.training.optimizer
            self.assertIsNotNone(
                optimizer.optimizer,
                'Should have created an optimizer')
            ref_parameters = experiment.parameters.get_optimizer_parameters()
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
                bricks = len(row.get_bricks())
                self.assertTrue(
                    is_valid_param_value(
                        experiment.parameters.get_layout_parameter('bricks'),
                        bricks),
                    'Invalid value')
                for brick in blueprint.layout.get_bricks():
                    blocks = len(brick.get_blocks())
                    self.assertTrue(
                        is_valid_param_value(
                            experiment.parameters.get_layout_parameter('blocks'),
                            blocks),
                        'Invalid value')

    def test_predefined_parameters(self):
        layout_definition = LayoutDefinition(
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
            layout_definition,
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
        layout_definition = LayoutDefinition(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block_template=[
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
            layout_definition,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None)
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
            for row in blueprint.layout.get_rows():
                bricks = len(row.get_bricks())
                self.assertTrue(
                    is_valid_param_value(
                        experiment.parameters.get_layout_parameter('bricks'),
                        bricks),
                    'Invalid value')
                for brick in blueprint.layout.get_bricks():
                    blocks = len(brick.get_blocks())
                    self.assertTrue(
                        is_valid_param_value(
                            experiment.parameters.get_layout_parameter('blocks'),
                            blocks),
                        'Invalid value')

                    for block in brick.get_blocks():
                        self.assertEqual(
                            len(layout_definition.block_template),
                            len(block.layers),
                            'Should have used template')

                        for i, layer in enumerate(layout_definition.block_template):
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
                                    'Should have used the predefined parameter value')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
