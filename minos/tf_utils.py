'''
Created on Mar 7, 2017

@author: julien
'''
import logging


def cpu_device():
    return '/cpu:0'


def get_available_gpus():
    try:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    except Exception as ex:
        logging.error(
            'Error while trying to list available GPUs: %s' % str(ex))
        return list()


def default_device():
    gpus = get_available_gpus()
    if len(gpus) > 0:
        return gpus[1]
    return cpu_device()


def get_logical_device(physical_device):
    if is_gpu_device(physical_device):
        return '/gpu:0'
    return physical_device


def is_cpu_device(device):
    return device\
        and isinstance(device, str)\
        and device.startswith('/cpu')


def is_gpu_device(device):
    return device\
        and isinstance(device, str)\
        and device.startswith('/gpu')


def get_device_idx(device):
    return device.split(':')[1]


def setup_tf_session(self, device):
    import tensorflow as tf
    config = tf.ConfigProto()
    if hasattr(config, 'gpu_options'):
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
    if is_gpu_device(device):
        config.gpu_options.visible_device_list = str(
            get_device_idx(device))
    elif is_cpu_device(device):
        config.gpu_options.visible_device_list = ''
    from keras import backend
    backend.set_session(tf.Session(config=config))
