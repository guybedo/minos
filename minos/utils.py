'''
Created on Feb 12, 2017

@author: julien
'''
from logging.config import dictConfigClass
from os.path import isfile
from posix import remove
import sys

from keras.models import load_model

from minos.model.parameters import get_custom_layers, get_custom_activations
from minos.tf_utils import setup_tf_session


def disable_sysout():
    sys.stdout.write = lambda s: s


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
