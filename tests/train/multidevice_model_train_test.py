'''
Created on Feb 18, 2017

@author: julien
'''
from genericpath import isfile
from os.path import join
import tempfile
import unittest

from minos.experiment.experiment import ExperimentParameters, Experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.design import create_random_blueprint
from minos.model.model import Layout, Objective, Optimizer, Metric
from minos.tf_utils import cpu_device
from minos.train.trainer import ModelTrainer
from minos.train.utils import CpuEnvironment
from minos.utils import disable_sysout
from tests.fixtures import get_reuters_dataset


class TrainTest(unittest.TestCase):

    def test_model_trainer(self):
        disable_sysout()
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_size = 50
            batch_iterator, test_batch_iterator, nb_classes = get_reuters_dataset(batch_size, 1000)
            layout = Layout(
                input_size=1000,
                output_size=nb_classes,
                output_activation='softmax')
            training = Training(
                objective=Objective('categorical_crossentropy'),
                optimizer=Optimizer(optimizer='Adam'),
                metric=Metric('categorical_accuracy'),
                stopping=EpochStoppingCondition(10),
                batch_size=batch_size)
            experiment_parameters = ExperimentParameters(use_default_values=True)
            experiment_parameters.layout_parameter('rows', 1)
            experiment_parameters.layout_parameter('blocks', 1)
            experiment_parameters.layout_parameter('layers', 1)
            experiment = Experiment(
                'test__reuters_experiment',
                layout,
                training,
                batch_iterator,
                test_batch_iterator,
                CpuEnvironment(n_jobs=1, data_dir=tmp_dir),
                parameters=experiment_parameters)

            blueprint = create_random_blueprint(experiment)
            trainer = ModelTrainer(batch_iterator, test_batch_iterator)
            model_filename = join(tmp_dir, 'model')
            model, history, _duration = trainer.train(
                blueprint,
                [cpu_device(), cpu_device()],
                save_best_model=True,
                model_filename=model_filename)
            model.predict(test_batch_iterator.X[0], len(test_batch_iterator.X[0]))
            self.assertIsNotNone(
                model,
                'should have fit the model')
            self.assertTrue(isfile(model_filename), 'Should have saved the model')
            self.assertIsNotNone(history, 'Should have the training history')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
