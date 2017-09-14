'''
Created on Feb 12, 2017

@author: julien
'''
from logging.config import dictConfigClass
from os.path import isfile
from posix import remove
import sys

import logging
from keras.models import load_model

from minos.model.parameters import get_custom_layers, get_custom_activations
from minos.tf_utils import setup_tf_session, cpu_device


def disable_sysout():
    sys.stdout.write = lambda s: s


def load_best_model(experiment_label, step, X_train=None, y_train=None, X_test=None, y_test=None, batch_size=1,
                    epochs=100):
    """ Here we load the blueprints generated during an experiment
    and create the Keras model from the top scoring blueprint
    """
    from minos.experiment.experiment import load_experiment_blueprints
    from minos.train.utils import Environment
    from minos.model.build import ModelBuilder

    blueprints = load_experiment_blueprints(
        experiment_label,
        step,
        Environment())
    model = ModelBuilder().build(
        blueprints[-1],
        cpu_device(),
        compile_model=True)
    logging.info(model.to_json())
    model.summary()
    if X_train is not None and len(X_train):
        logging.info("Now training model from blueprint")
        model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2, epochs=epochs)

    return model


def load_keras_model(filename, device=None):
    setup_tf_session(device)
    custom_objects = dict()
    custom_objects.update(get_custom_activations())
    custom_objects.update({
        layer[0].__name__: layer[0]
        for _name, layer in get_custom_layers().items()})
    return load_model(filename, custom_objects=custom_objects)


def setup_logging(filename, level='INFO', resume=False):
    if not resume and isfile(filename):
        remove(filename)
    print('Logging to %s, level=%s' % (filename, level))
    logging_configuration = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'file': {
                'level': level,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': filename
            }
        },
        'loggers': {
            '': {
                'handlers': ['file'],
                'level': level,
                'propagate': True
            }
        }
    }
    dictConfigClass(logging_configuration).configure()


def setup_console_logging(level='INFO'):
    print('Logging to console, level=%s' % level)
    logging_configuration = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            }
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': level,
                'propagate': True
            }
        }
    }
    dictConfigClass(logging_configuration).configure()
