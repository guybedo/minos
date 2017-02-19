'''
Created on Feb 8, 2017

@author: julien
'''


class Training(object):

    def __init__(self, objective, optimizer, metric, stopping, batch_size):
        self.objective = objective
        self.optimizer = optimizer
        self.metric = metric
        self.stopping = stopping
        self.batch_size = batch_size

    def todict(self):
        return dict(self.__dict__)


class EpochStoppingCondition(object):

    def __init__(self, epoch):
        self.epoch = epoch


class MetricDecreaseStoppingCondition(object):

    def __init__(self, noprogress_count=3, min_epoch=0, max_epoch=0):
        self.patience = noprogress_count
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.monitor = 'val_accuracy'
        self.epoch = 0

    def is_at_least_min_epoch(self):
        return not self.min_epoch\
            or self.step >= self.min_epoch

    def is_at_most_max_epoch(self):
        return not self.max_epoch\
            or self.epoch <= self.max_epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if not self.is_at_least_min_epoch():
            return
        if not self.is_at_most_max_epoch():
            self.stopped_epoch = epoch
            self.model.stop_training = True
            return
        super().on_epoch_end(epoch, logs)
