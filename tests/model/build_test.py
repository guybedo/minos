'''
Created on Feb 15, 2017

@author: julien
'''
from copy import deepcopy
import unittest

from keras.layers.core import Dense

from minos.experiment.experiment import Experiment, ExperimentParameters,\
    check_experiment_parameters
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.design import create_random_blueprint, mix_blueprints,\
    mutate_blueprint
from minos.model.model import Layout, Objective, Metric
from minos.model.parameter import int_param, string_param, float_param
from minos.model.parameters import register_custom_activation,\
    register_custom_layer, reference_parameters
from minos.train.utils import cpu_device


class BuildTest(unittest.TestCase):

    def test_build(self):
        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax')
        training = Training(
            objective=Objective('categorical_crossentropy'),
            optimizer=None,
            metric=Metric('categorical_accuracy'),
            stopping=EpochStoppingCondition(10),
            batch_size=250)

        experiment_parameters = ExperimentParameters(use_default_values=False)
        experiment_parameters.layout_parameter('blocks', int_param(1, 5))
        experiment_parameters.layout_parameter('layers', int_param(1, 5))
        experiment_parameters.layer_parameter('Dense.output_dim', int_param(10, 500))
        experiment_parameters.layer_parameter('Dense.activation', string_param(['relu', 'tanh']))
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
        for _ in range(5):
            blueprint1 = create_random_blueprint(experiment)
            model = ModelBuilder().build(blueprint1, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint2 = create_random_blueprint(experiment)
            model = ModelBuilder().build(blueprint2, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint3 = mix_blueprints(blueprint1, blueprint2, experiment_parameters)
            model = ModelBuilder().build(blueprint3, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint4 = mutate_blueprint(blueprint1, experiment_parameters, mutate_in_place=False)
            model = ModelBuilder().build(blueprint4, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')

    def test_build_w_custom_definitions(self):

        def custom_activation(x):
            return x

        register_custom_activation('custom_activation', custom_activation)
        register_custom_layer(
            'Dense2',
            Dense,
            deepcopy(reference_parameters['layers']['Dense']),
            True)

        layout = Layout(
            input_size=100,
            output_size=10,
            output_activation='softmax',
            block=['Dense2'])
        training = Training(
            objective=Objective('categorical_crossentropy'),
            optimizer=None,
            metric=Metric('categorical_accuracy'),
            stopping=EpochStoppingCondition(5),
            batch_size=250)

        experiment_parameters = ExperimentParameters(use_default_values=False)
        experiment_parameters.layout_parameter('blocks', int_param(1, 5))
        experiment_parameters.layout_parameter('layers', int_param(1, 5))
        experiment_parameters.layout_parameter('layer.type', string_param(['Dense2']))
        experiment_parameters.layer_parameter('Dense2.output_dim', int_param(10, 500))
        experiment_parameters.layer_parameter('Dense2.activation', string_param(['custom_activation']))
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
        for _ in range(5):
            blueprint1 = create_random_blueprint(experiment)
            for layer in blueprint1.layout.get_layers():
                self.assertEqual('Dense2', layer.layer_type, 'Should have used custom layer')
            model = ModelBuilder().build(blueprint1, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint2 = create_random_blueprint(experiment)
            for layer in blueprint2.layout.get_layers():
                self.assertEqual('Dense2', layer.layer_type, 'Should have used custom layer')
            model = ModelBuilder().build(blueprint2, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint3 = mix_blueprints(blueprint1, blueprint2, experiment_parameters)
            for layer in blueprint3.layout.get_layers():
                self.assertEqual('Dense2', layer.layer_type, 'Should have used custom layer')
            model = ModelBuilder().build(blueprint3, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')
            blueprint4 = mutate_blueprint(blueprint1, experiment_parameters, mutate_in_place=False)
            for layer in blueprint4.layout.get_layers():
                self.assertEqual('Dense2', layer.layer_type, 'Should have used custom layer')
            model = ModelBuilder().build(blueprint4, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
