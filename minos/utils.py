'''
Created on Feb 12, 2017

@author: julien
'''
from logging.config import dictConfigClass


def setup_logging(filename, level='INFO'):
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
