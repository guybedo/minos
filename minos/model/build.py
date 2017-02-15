'''
Created on Feb 6, 2017

@author: julien
'''
import keras
from keras.models import Sequential

import tensorflow as tf


class ModelBuilder(object):

    def __init__(self):
        pass

    def build(self, blueprint, device):
        model = self._build_blueprint(blueprint)
        model.compile(
            optimizer=_build_optimizer(blueprint.training),
            loss=blueprint.training.loss.loss,
            metrics=[blueprint.training.metric.metric])
        return model

    def _build_blueprint(self, blueprint, device):
        with tf.device(device):
            return Sequential()


def _build_optimizer(training):
    optimizer = getattr(keras.optimizers, training.optimizer.optimizer)
    return optimizer(**training.optimizer.parameters)
