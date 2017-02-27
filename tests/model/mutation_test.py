'''
Created on Feb 15, 2017

@author: julien
'''
from copy import deepcopy
import unittest

from keras import activations, backend
from keras.engine.topology import Layer
from keras.layers.core import Dense

from minos.experiment.experiment import Experiment, ExperimentParameters,\
    check_experiment_parameters
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.design import create_random_blueprint, mutate_blueprint,\
    mix_blueprints
from minos.model.model import Layout, Objective, Metric
from minos.model.parameter import int_param, string_param, float_param
from minos.model.parameters import register_custom_activation,\
    register_custom_layer, reference_parameters
from minos.train.utils import cpu_device


class MutationTest(unittest.TestCase):

    def test_mutate_layout(self):
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
            mutant = mutate_blueprint(
                blueprint,
                parameters=experiment.parameters,
                p_mutate_layout=1,
                layout_mutation_count=1,
                layout_mutables=['rows'],
                mutate_in_place=False)
            self.assertTrue(
                len(mutant.layout.rows) != len(blueprint.layout.rows),
                'Should have mutated rows')
            mutant = mutate_blueprint(
                blueprint,
                parameters=experiment.parameters,
                p_mutate_layout=1,
                layout_mutation_count=1,
                layout_mutables=['blocks'],
                mutate_in_place=False)
            self.assertTrue(
                len(mutant.layout.get_blocks()) != len(blueprint.layout.get_blocks()),
                'Should have mutated blocks')
            mutant = mutate_blueprint(
                blueprint,
                parameters=experiment.parameters,
                p_mutate_layout=1,
                layout_mutation_count=1,
                layout_mutables=['layers'],
                mutate_in_place=False)
            self.assertTrue(
                len(mutant.layout.get_layers()) != len(blueprint.layout.get_layers()),
                'Should have mutated layers')

    def test_mutate_parameters(self):
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
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            mutant = mutate_blueprint(
                blueprint,
                parameters=experiment.parameters,
                p_mutate_layout=0,
                p_mutate_param=1,
                mutate_in_place=False)

            for row_idx, row in enumerate(mutant.layout.rows):
                for block_idx, block in enumerate(row.blocks):
                    for layer_idx, layer in enumerate(block.layers):
                        original_row = blueprint.layout.rows[row_idx]
                        original_block = original_row.blocks[block_idx]
                        original_layer = original_block.layers[layer_idx]
                        for name, value in layer.parameters.items():
                            if value == original_layer.parameters[name]:
                                pass
                            self.assertTrue(
                                value != original_layer.parameters[name],
                                'Should have mutated parameter')

    def test_mutate_w_custom_definitions(self):

        def custom_activation(x):
            return x

        register_custom_activation('custom_activation', custom_activation)
        register_custom_layer('Dense2', Dense, deepcopy(reference_parameters['layers']['Dense']))

        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block=['Dense', 'Dense2'])
        training = Training(
            objective=Objective('categorical_crossentropy'),
            optimizer=None,
            metric=Metric('categorical_accuracy'),
            stopping=EpochStoppingCondition(5),
            batch_size=250)

        experiment_parameters = ExperimentParameters(use_default_values=False)
        experiment_parameters.layout_parameter('blocks', int_param(1, 5))
        experiment_parameters.layout_parameter('layers', int_param(1, 5))
        experiment_parameters.layer_parameter('Dense2.output_dim', int_param(10, 500))
        experiment_parameters.layer_parameter('Dropout.p', float_param(0.1, 0.9))
        experiment_parameters.all_search_parameters(True)
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None,
            parameters=experiment_parameters)
        check_experiment_parameters(experiment)
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            mutant = mutate_blueprint(
                blueprint,
                parameters=experiment.parameters,
                p_mutate_layout=0,
                p_mutate_param=1,
                mutate_in_place=False)
            for row_idx, row in enumerate(mutant.layout.rows):
                for block_idx, block in enumerate(row.blocks):
                    for layer_idx, layer in enumerate(block.layers):
                        original_row = blueprint.layout.rows[row_idx]
                        original_block = original_row.blocks[block_idx]
                        original_layer = original_block.layers[layer_idx]
                        for name, value in layer.parameters.items():
                            self.assertTrue(
                                value != original_layer.parameters[name],
                                'Should have mutated parameter')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
