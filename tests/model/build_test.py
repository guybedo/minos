'''
Created on Feb 15, 2017

@author: julien
'''
import unittest

from minos.experiment.experiment import Experiment, ExperimentParameters
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.design import create_random_blueprint
from minos.model.model import Layout, Objective, Metric
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
        experiment = Experiment(
            'test',
            layout,
            training,
            batch_iterator=None,
            test_batch_iterator=None,
            environment=None,
            parameters=experiment_parameters)
        for _ in range(10):
            blueprint = create_random_blueprint(experiment)
            model = ModelBuilder().build(blueprint, cpu_device())
            self.assertIsNotNone(model, 'Should have built a model')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
