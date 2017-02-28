'''
Created on Feb 18, 2017

@author: julien
'''
from genericpath import isfile
from os.path import join
import tempfile
import unittest

from minos.experiment.experiment import ExperimentParameters, Experiment,\
    _assert_valid_training_parameters
from minos.experiment.training import Training, EpochStoppingCondition,\
    AccuracyDecreaseStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.design import create_random_blueprint
from minos.model.model import Layout, Objective, Optimizer, Metric
from minos.train.trainer import ModelTrainer
from minos.train.utils import default_device, CpuEnvironment, cpu_device
from minos.utils import disable_sysout
from tests.fixtures import get_reuters_dataset


class TrainTest(unittest.TestCase):

    def test_train(self):
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
            model = ModelBuilder().build(blueprint, default_device())
            result = model.fit_generator(
                generator=batch_iterator,
                samples_per_epoch=batch_iterator.samples_per_epoch,
                nb_epoch=10,
                validation_data=test_batch_iterator,
                nb_val_samples=test_batch_iterator.sample_count)
            self.assertIsNotNone(
                result,
                'should have fit the model')
            score = model.evaluate_generator(
                test_batch_iterator,
                val_samples=test_batch_iterator.sample_count)
            self.assertIsNotNone(
                score,
                'should have evaluated the model')

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
                cpu_device(),
                save_best_model=True,
                model_filename=model_filename)
            self.assertIsNotNone(
                model,
                'should have fit the model')
            self.assertTrue(isfile(model_filename), 'Should have saved the model')
            self.assertIsNotNone(history, 'Should have the training history')

    def test_early_stopping_condition_test(self):
        disable_sysout()
        with tempfile.TemporaryDirectory() as tmp_dir:
            batch_size = 50
            min_epoch = 10
            max_epoch = 15
            batch_iterator, test_batch_iterator, nb_classes = get_reuters_dataset(batch_size, 1000)
            layout = Layout(
                input_size=1000,
                output_size=nb_classes,
                output_activation='softmax')
            training = Training(
                objective=Objective('categorical_crossentropy'),
                optimizer=Optimizer(optimizer='Adam'),
                metric=Metric('categorical_accuracy'),
                stopping=AccuracyDecreaseStoppingCondition(
                    metric='categorical_accuracy',
                    noprogress_count=2,
                    min_epoch=min_epoch,
                    max_epoch=max_epoch),
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
            _assert_valid_training_parameters(experiment)

            blueprint = create_random_blueprint(experiment)
            trainer = ModelTrainer(batch_iterator, test_batch_iterator)
            model, history, _duration = trainer.train(blueprint, cpu_device(), save_best_model=False)
            self.assertTrue(len(history.epoch) >= min_epoch, 'Should have trained for at least min epoch')
            self.assertTrue(len(history.epoch) <= max_epoch, 'Should have trained for max epoch')
            self.assertIsNotNone(
                model,
                'should have fit the model')
            score = model.evaluate_generator(
                test_batch_iterator,
                val_samples=test_batch_iterator.sample_count)
            self.assertIsNotNone(
                score,
                'should have evaluated the model')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
