'''
Created on Feb 12, 2017

@author: julien
'''
import logging
from multiprocessing import Queue, Process
import numpy
from threading import Thread
from time import time
import traceback

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from minos.experiment.training import EpochStoppingCondition,\
    AccuracyDecreaseStoppingCondition, AccuracyDecreaseStoppingConditionWrapper
from minos.train.utils import is_gpu_device, get_device_idx, get_logical_device
from minos.utils import disable_sysout


class MultiProcessModelTrainer(object):

    def __init__(self, batch_iterator, test_batch_iterator, environment):
        self.batch_iterator = batch_iterator
        self.test_batch_iterator = test_batch_iterator
        self.environment = environment

    def build_and_train_models(self, blueprints):
        logging.debug('Training %d models' % len(blueprints))
        return self._start_training_workers(blueprints)

    def _start_training_workers(self, blueprints):
        try:
            total_n_jobs = sum(self.environment.n_jobs)
            work_queue = Queue(total_n_jobs)
            result_queue = Queue(total_n_jobs)
            self.processes = [
                Process(
                    target=model_training_worker,
                    args=(
                        self.batch_iterator,
                        self.test_batch_iterator,
                        device_id,
                        device,
                        work_queue,
                        result_queue))
                for device_id, device in enumerate(self.environment.devices)
                for _job in range(self.environment.n_jobs[device_id])]
            self.process_count = 0
            for process in self.processes:
                self.process_count += 1
                process.start()

            def _work_feeder():
                count = len(blueprints)
                for i, blueprint in enumerate(blueprints):
                    work_queue.put((i, count, blueprint))
                for _ in range(sum(self.environment.n_jobs)):
                    work_queue.put(None)
            Thread(target=_work_feeder).start()

            results = []
            while self.process_count > 0:
                result = result_queue.get()
                if result:
                    logging.debug(
                        'Blueprint %d: score %f after %d epochs',
                        result[0],
                        result[1],
                        result[2])
                    results.append(result)
                else:
                    self.process_count -= 1
            results = list(
                sorted(
                    results,
                    key=lambda e: e[0]))
            return results
        except Exception as ex:
            logging.error(ex)
            logging.error(traceback.format_exc())


class ModelTrainer(object):

    def __init__(self, batch_iterator, test_batch_iterator):
        from minos.model.build import ModelBuilder
        self.model_builder = ModelBuilder()
        self.batch_iterator = batch_iterator
        self.test_batch_iterator = test_batch_iterator

    def train(self, blueprint, device,
              save_best_model=False, model_filename=None):
        try:
            model = self.model_builder.build(
                blueprint,
                get_logical_device(device))
        except Exception as ex:
            return None, None, 0
        try:
            self._setup_tf(device)
            nb_epoch, callbacks = self._get_stopping_parameters(blueprint)
            if save_best_model:
                callbacks.append(self._get_model_save_callback(
                    model_filename,
                    blueprint.training.metric.metric))
            start = time()
            history = model.fit_generator(
                self.batch_iterator,
                self.batch_iterator.samples_per_epoch,
                nb_epoch,
                callbacks=callbacks,
                validation_data=self.test_batch_iterator,
                nb_val_samples=self.test_batch_iterator.sample_count)
            if save_best_model:
                del model
                model = load_model(model_filename)
            return model, history, (time() - start)
        except Exception as ex:
            logging.error(ex)
            logging.error(traceback.format_exc())
        return None, None, 0

    def _get_model_save_callback(self, model_filename, metric):
        checkpoint = ModelCheckpoint(
            model_filename,
            monitor=metric,
            save_best_only=True)
        return checkpoint

    def _get_stopping_parameters(self, blueprint):
        if isinstance(blueprint.training.stopping, EpochStoppingCondition):
            nb_epoch = blueprint.training.stopping.epoch
            stopping_callbacks = []
        if isinstance(blueprint.training.stopping, AccuracyDecreaseStoppingCondition):
            nb_epoch = max(
                1,
                blueprint.training.stopping.min_epoch,
                blueprint.training.stopping.max_epoch)
            stopping_callbacks = [
                AccuracyDecreaseStoppingConditionWrapper(blueprint.training.stopping)]
        return nb_epoch, stopping_callbacks

    def _setup_tf(self, device):
        import tensorflow as tf
        config = tf.ConfigProto()
        if is_gpu_device(device):
            config.allow_soft_placement = True
            config.gpu_options.visible_device_list = str(
                get_device_idx(device))
            config.gpu_options.allow_growth = True
        elif hasattr(config, 'gpu_options'):
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = ''
        from keras import backend
        backend.set_session(tf.Session(config=config))


def model_training_worker(batch_iterator, test_batch_iterator,
                          device_id, device, work_queue, result_queue):
    disable_sysout()
    model_trainer = ModelTrainer(
        batch_iterator,
        test_batch_iterator)
    work = work_queue.get()
    while work:
        try:
            idx, _total, blueprint = work
            _model, history, duration = model_trainer.train(
                blueprint,
                device)
            epoch = numpy.argmax(history.history[blueprint.training.metric.metric])
            score = history.history[blueprint.training.metric.metric][epoch]
            result_queue.put((idx, score, epoch, blueprint, duration, device_id))
            work = work_queue.get()
        except Exception as ex:
            logging.error(ex)
    result_queue.put(None)
