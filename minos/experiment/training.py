'''
Created on Feb 8, 2017

@author: julien
'''
from keras.callbacks import EarlyStopping


class Training(object):

    def __init__(self, objective, optimizer, metric, stopping, batch_size):
        self.objective = objective
        self.optimizer = optimizer
        self.metric = metric
        self.stopping = stopping
        self.batch_size = batch_size

    def todict(self):
        return {
            'objective': self.objective.todict(),
            'optimizer': self.optimizer.todict(),
            'metric': self.metric.todict(),
            'stopping': self.stopping.todict(),
            'batch_size': self.batch_size}


class EpochStoppingCondition(object):
    """ Stop training after a fixed number of epochs
        # Arguments
        epoch: number of epochs after which training will be stopped.
    """

    def __init__(self, epoch):
        self.epoch = epoch

    def todict(self):
        return dict(vars(self))


class AccuracyDecreaseStoppingCondition(object):
    """ Stop training when the metric observed has stopped improving.
        # Arguments
        noprogress_count: number of epochs with no improvement
            after which training will be stopped.
        min_epoch: minimum number of epochs.
        max_epoch: maximum number of epochs.
    """

    def __init__(self, noprogress_count=3, min_epoch=0, max_epoch=0):
        super().__init__(
            monitor='val_accuracy',
            patience=noprogress_count)
        self.noprogress_count = noprogress_count
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch = 0

    def is_at_least_min_epoch(self):
        return not self.min_epoch\
            or self.step >= self.min_epoch

    def is_at_most_max_epoch(self):
        return not self.max_epoch\
            or self.epoch <= self.max_epoch

    def todict(self):
        return dict(vars(self))


class AccuracyDecreaseStoppingConditionWrapper(EarlyStopping):

    def __init(self, accuracy_condition):
        super().__init__(
            monitor='val_accuracy',
            patience=accuracy_condition.noprogress_count)
        self.accuracy_condition = accuracy_condition

    def on_epoch_end(self, epoch, logs=None):
        self.accuracy_condition.epoch = epoch
        if not self.accuracy_condition.is_at_least_min_epoch():
            return
        if not self.accuracy_condition.is_at_most_max_epoch():
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return
        super().on_epoch_end(epoch, logs)
