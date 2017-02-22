'''
Created on Feb 12, 2017

@author: julien
'''
import logging
import numpy
from os import path, makedirs
from os.path import join
from posix import access, W_OK
from random import Random
import traceback

from tensorflow.python.client import device_lib

import tensorflow as tf


random = Random()


def cpu_device():
    return '/cpu:0'


def get_available_gpus():
    try:
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


def is_gpu_device(device):
    return device.startswith('/gpu')


def get_device_idx(device):
    return device.split(':')[1]


class Environment(object):

    def __init__(self, devices=None, n_jobs=None,
                 data_dir=None, tf_logging_level=tf.logging.ERROR):
        self.devices = devices
        self.n_jobs = n_jobs
        if devices and n_jobs and not isinstance(n_jobs, list):
            self.n_jobs = [n_jobs for _ in devices]
        self.data_dir = data_dir or self._init_minos_dir()
        self.tf_logging_level = tf_logging_level

    def _init_minos_dir(self):
        base_dir = path.expanduser('~')
        if not access(base_dir, W_OK):
            base_dir = path.join('/tmp')
        minos_dir = join(base_dir, 'minos')
        if not path.exists(minos_dir):
            makedirs(minos_dir)
        return minos_dir


class CpuEnvironment(Environment):

    def __init__(self, n_jobs=1, data_dir=None,
                 tf_logging_level=tf.logging.ERROR):
        super().__init__(
            ['/cpu:0'],
            n_jobs,
            data_dir=data_dir,
            tf_logging_level=tf_logging_level)


class GpuEnvironment(Environment):

    def __init__(self, devices=None, n_jobs=1, data_dir=None,
                 tf_logging_level=tf.logging.ERROR):
        super().__init__(
            devices or get_available_gpus(),
            n_jobs,
            data_dir=data_dir,
            tf_logging_level=tf_logging_level)


class SimpleBatchIterator(object):

    def __init__(self, X, y, batch_size,
                 X_transform=None, y_transform=None,
                 autoloop=False, autorestart=False, preload=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.X_transform = X_transform
        self.y_transform = y_transform
        self.autoloop = autoloop
        self.autorestart = autorestart
        self.index = 0
        self.preload = preload
        self.batch_size = batch_size
        self.sample_count = len(X)
        self.samples_per_epoch = self.sample_count
        self.X, self.y = create_batches(
            X,
            y,
            batch_size,
            shuffle=True)

    def _transform_data(self, X, y):
        if self.X_transform:
            X = self.X_transform.fit_transform(X)
        if self.y_transform:
            y = self.y_transform.fit_transform(y)
        return numpy.asarray(X), numpy.asarray(y)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.index >= len(self.X):
                if self.autorestart or self.autoloop:
                    self.index = 0
                if self.autorestart or not self.autoloop:
                    return None
            X, y = self._transform_data(
                self.X[self.index],
                self.y[self.index])
            self.index += 1
            return X, y
        except Exception as ex:
            logging.error('Error while iterating %s' % str(ex))
            try:
                logging.error(traceback.format_exc())
            finally:
                pass
            raise ex


def create_batches(X, y, batch_size, shuffle=True):
    X = [
        X[i:i + batch_size]
        for i in range(0, len(X), batch_size)]
    y = [
        y[i:i + batch_size]
        for i in range(0, len(y), batch_size)]
    if shuffle:
        shuffle_batch(X, y)
    return X, y


def shuffle_batches(X_batches, y_batches):
    for X, y in zip(X_batches, y_batches):
        shuffle_batch(X, y)


def shuffle_batch(X, y):
    for i in range(len(X)):
        swap_idx = random.randint(i, len(X) - 1)
        swap(X, i, swap_idx)
        swap(y, i, swap_idx)


def swap(values, idx1, idx2):
    swap = values[idx2]
    values[idx2] = values[idx1]
    values[idx1] = swap
