'''
Created on Feb 15, 2017

@author: julien
'''
import unittest

from minos.experiment.experiment import Experiment, ExperimentParameters,\
    check_experiment_parameters
from minos.experiment.training import Training
from minos.model.design import create_random_blueprint, mix_blueprints
from minos.model.model import Layout


class MixTest(unittest.TestCase):

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
            blueprint1 = create_random_blueprint(experiment)
            blueprint2 = create_random_blueprint(experiment)
            mutant = mix_blueprints(
                blueprint1,
                blueprint2,
                parameters=experiment.parameters,
                p_mutate_param=0.1)
            blueprints = [blueprint1, blueprint2]
            parent_rows = [
                len(b.layout.rows) for b in blueprints]
            self.assertTrue(
                len(mutant.layout.rows) in parent_rows,
                'Should have used one of the parents')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
