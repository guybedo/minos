'''
Created on Feb 6, 2017

@author: julien
'''
import numpy
from os.path import join
import tempfile


from examples.ga.dataset import get_reuters_dataset
from minos.experiment.experiment import Experiment, ExperimentParameters
from minos.experiment.training import Training, AccuracyDecreaseStoppingCondition,\
    get_associated_validation_metric
from minos.model.design import create_random_blueprint
from minos.model.model import Objective, Optimizer, Metric, Layout
from minos.train.trainer import ModelTrainer
from minos.utils import setup_console_logging
import numpy as np


np.random.seed(1337)


def create_experiment(input_size, output_size, batch_size):
    training = Training(
        objective=Objective('categorical_crossentropy'),
        optimizer=Optimizer(optimizer='Adam'),
        metric=Metric('categorical_accuracy'),
        stopping=AccuracyDecreaseStoppingCondition(
            metric='categorical_accuracy',
            min_epoch=5,
            max_epoch=25,
            noprogress_count=5),
        batch_size=batch_size)
    parameters = ExperimentParameters(use_default_values=True)
    layout = Layout(
        input_size=input_size,
        output_size=output_size,
        output_activation='softmax')
    experiment = Experiment(
        label='reuters_train_multi_gpu',
        layout=layout,
        training=training,
        parameters=parameters)
    return experiment

def train_multi_gpu(max_words = 1000, batch_size=32):
    batch_iterator, test_batch_iterator, nb_classes = get_reuters_dataset(batch_size, max_words)
    experiment = create_experiment(max_words, nb_classes, batch_size)
    blueprint = create_random_blueprint(experiment)
    devices = ['/gpu:0', '/gpu:1']
    trainer = ModelTrainer(batch_iterator, test_batch_iterator)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model, history, _duration = trainer.train(
            blueprint,
            devices,
            save_best_model=True,
            model_filename=join(tmp_dir, 'model'))
        metric = get_associated_validation_metric(blueprint.training.metric.metric)
        epoch = numpy.argmax(history.history[metric])
        score = history.history[metric][epoch]
        print('Final training score %r after %d epoch' % (score, epoch))
        
        test_size = 10
        y_true = numpy.argmax(test_batch_iterator.y[0][:test_size], axis=1)
        y_pred = numpy.argmax(model.predict(test_batch_iterator.X[0][:test_size]), axis=1)
        evaluation = numpy.mean(y_true==y_pred)
        print('Final evaluation score %f' % evaluation)
        print('Predictions (true, pred) %r' % list(zip(y_true.tolist(), y_pred.tolist())))
        
def main():
    setup_console_logging('DEBUG')
    train_multi_gpu()

if __name__ == '__main__':
    main()
