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

    def __init__(self, max_epoch, metric='accuracy',
                 noprogress_count=3, min_epoch=0):
        self.metric = metric
        self.noprogress_count = noprogress_count
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch = 0

    def is_min_epoch_defined(self):
        return self.min_epoch and self.min_epoch > 0

    def is_max_epoch_defined(self):
        return self.max_epoch and self.max_epoch > 0

    def is_at_least_min_epoch(self):
        return self.is_min_epoch_defined()\
            and self.epoch >= self.min_epoch

    def is_at_most_max_epoch(self):
        return self.is_max_epoch_defined()\
            and self.epoch <= self.max_epoch

    def todict(self):
        return dict(vars(self))


def get_associated_validation_metric(metric):
    if not metric:
        return None
    if metric.startswith('val_'):
        return metric
    return 'val_%s' % metric


class AccuracyDecreaseStoppingConditionWrapper(EarlyStopping):

    def __init__(self, accuracy_condition):
        super().__init__(
            monitor=get_associated_validation_metric(accuracy_condition.metric),
            patience=accuracy_condition.noprogress_count)
        self.accuracy_condition = accuracy_condition

    def on_epoch_end(self, epoch, logs=None):
        self.accuracy_condition.epoch = epoch
        if self.accuracy_condition.is_min_epoch_defined()\
                and not self.accuracy_condition.is_at_least_min_epoch():
            return
        if self.accuracy_condition.is_max_epoch_defined()\
                and not self.accuracy_condition.is_at_most_max_epoch():
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return
        super().on_epoch_end(epoch, logs)
