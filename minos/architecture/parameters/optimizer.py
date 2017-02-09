'''
Created on Feb 8, 2017

@author: julien
'''
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax,\
    Nadam
from minos.architecture.parameters.parameter import float_param, boolean_param


optimizers = {
    SGD: {
        'lr': float_param(default=1e-3),
        'momentum': float_param(),
        'decay': float_param(),
        'nesterov': boolean_param()},
    RMSprop: {
        'lr': float_param(default=1e-3),
        'rho': float_param(default=0.9),
        'epsilon': float_param(default=1e-08),
        'decay': float_param()},
    Adagrad: {
        'lr': float_param(default=1e-3),
        'epsilon': float_param(default=1e-08),
        'decay': float_param()},
    Adadelta: {
        'lr': float_param(default=1e-3),
        'rho': float_param(default=0.9),
        'epsilon': float_param(default=1e-08),
        'decay': float_param()},
    Adam: {
        'lr': float_param(default=1e-3),
        'beta_1': float_param(default=0.9),
        'beta_2': float_param(default=0.999),
        'epsilon': float_param(default=1e-08),
        'decay': float_param()},
    Adamax: {
        'lr': float_param(default=1e-3),
        'beta_1': float_param(default=0.9),
        'beta_2': float_param(default=0.999),
        'epsilon': float_param(default=1e-08),
        'decay': float_param()},
    Nadam: {
        'lr': float_param(default=1e-3),
        'beta_1': float_param(default=0.9),
        'beta_2': float_param(default=0.999),
        'epsilon': float_param(default=1e-08),
        'schedule_decay': float_param()}}
