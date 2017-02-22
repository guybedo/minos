'''
Created on Feb 12, 2017

@author: julien
'''
from logging.config import dictConfigClass
from os.path import isfile
from posix import remove
import sys


def disable_sysout():
    sys.stdout.write = lambda s: s


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
