'''
Created on Feb 6, 2017

@author: julien
'''
import keras
from keras.models import Sequential

from minos.train.utils import default_device
import tensorflow as tf


class ModelBuilder(object):

    def __init__(self):
        pass

    def build(self, blueprint, device=default_device()):
        model = _build_model(blueprint, device)
        model.compile(
            optimizer=_build_optimizer(blueprint.training),
            loss=blueprint.training.loss.loss,
            metrics=[blueprint.training.metric.metric])
        return model


def _build_model(self, blueprint, device=default_device()):
    with tf.device(device):
        return Sequential()


def _build_optimizer(training):
    optimizer = getattr(keras.optimizers, training.optimizer.optimizer)
    return optimizer(**training.optimizer.parameters)
