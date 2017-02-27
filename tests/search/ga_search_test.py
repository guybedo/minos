'''
Created on Feb 15, 2017

@author: julien
'''
from os.path import isfile
import tempfile
import unittest

from minos.experiment.experiment import Experiment, ExperimentParameters,\
    load_experiment_blueprints, load_experiment_checkpoint,\
    load_experiment_best_blueprint
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.model import Layout, Objective, Metric, Optimizer
from minos.model.parameter import int_param
from minos.train.utils import CpuEnvironment, cpu_device, Environment
from tests.fixtures import get_reuters_dataset
from minos.utils import disable_sysout


class GaSearchTest(unittest.TestCase):

    def test_ga_search(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            epoch = 3
            generations = 2
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
                stopping=EpochStoppingCondition(epoch),
                batch_size=batch_size)
            experiment_parameters = ExperimentParameters(use_default_values=False)
            experiment_parameters.layout_parameter('rows', 1)
            experiment_parameters.layout_parameter('blocks', 1)
            experiment_parameters.layout_parameter('layers', 1)
            experiment_parameters.layer_parameter('Dense.output_dim', int_param(10, 500))
            experiment_parameters.all_search_parameters(True)

            experiment_label = 'test__reuters_experiment'
            experiment = Experiment(
                experiment_label,
                layout,
                training,
                batch_iterator,
                test_batch_iterator,
                CpuEnvironment(n_jobs=2, data_dir=tmp_dir),
                parameters=experiment_parameters)
            run_ga_search_experiment(experiment, population_size=2, generations=2)
            self.assertTrue(
                isfile(experiment.get_log_filename()),
                'Should have logged')
            self.assertTrue(
                isfile(experiment.get_step_data_filename(0)),
                'Should have logged')
            self.assertTrue(
                isfile(experiment.get_step_log_filename(0)),
                'Should have logged')
            blueprints = load_experiment_blueprints(
                experiment_label, 0,
                Environment(data_dir=tmp_dir))
            self.assertTrue(
                len(blueprints) > 0,
                'Should have saved/loaded blueprints')
            model = ModelBuilder().build(
                blueprints[0],
                cpu_device())
            disable_sysout()
            model.fit_generator(
                generator=batch_iterator,
                samples_per_epoch=batch_iterator.samples_per_epoch,
                nb_epoch=5,
                validation_data=test_batch_iterator,
                nb_val_samples=test_batch_iterator.sample_count)
            score = model.evaluate_generator(
                test_batch_iterator,
                val_samples=test_batch_iterator.sample_count)
            self.assertTrue(score[1] > 0, 'Should have valid score')

            step, population = load_experiment_checkpoint(experiment)
            self.assertEqual(generations - 1, step, 'Should have loaded checkpoint')
            self.assertIsNotNone(population, 'Should have loaded checkpoint')
            blueprint = load_experiment_best_blueprint(
                experiment.label,
                environment=CpuEnvironment(n_jobs=2, data_dir=tmp_dir))
            model = ModelBuilder().build(
                blueprint,
                cpu_device(),
                compile_model=False)
            self.assertIsNotNone(model, 'Should have loaded and built best model from experiment')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
