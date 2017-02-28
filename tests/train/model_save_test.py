'''
Created on Feb 18, 2017

@author: julien
'''
from os.path import join
import tempfile
import unittest

from minos.experiment.experiment import ExperimentParameters, Experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.build import ModelBuilder
from minos.model.design import create_random_blueprint
from minos.model.model import Layout, Objective, Optimizer, Metric
from minos.model.parameter import int_param
from minos.model.parameters import register_custom_activation,\
    register_custom_layer
from minos.train.utils import default_device, CpuEnvironment
from minos.utils import disable_sysout, load_model, load_keras_model
from tests.fixtures import get_reuters_dataset, CustomLayer


class ModelSaveTest(unittest.TestCase):

    def test_save(self):
        disable_sysout()

        def custom_activation(x):
            return x

        register_custom_activation('custom_activation', custom_activation)
        register_custom_layer('custom_layer', CustomLayer, {'output_dim': int_param(1, 100)})

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
            experiment_parameters.layout_parameter('block.layer_type', 'custom_layer')
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
            model.fit_generator(
                generator=batch_iterator,
                samples_per_epoch=batch_iterator.samples_per_epoch,
                nb_epoch=10,
                validation_data=test_batch_iterator,
                nb_val_samples=test_batch_iterator.sample_count)
            filepath = join(tmp_dir, 'model')
            model.save(filepath)
            model = load_keras_model(filepath)
            self.assertIsNotNone(model, 'Should have loaded the model')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
