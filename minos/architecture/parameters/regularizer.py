'''
Created on Feb 8, 2017

@author: julien
'''

from keras.regularizers import l1, l2, l1l2, activity_l1, activity_l2,\
    activity_l1l2

from minos.architecture.parameters.parameter import float_param

regularizers = {
    l1: {
        'l': float_param},
    l2: {
        'l': float_param},
    l1l2: {
        'l1': float_param,
        'l2': float_param}}
activity_regularizers = {
    activity_l1: {
        'l': float_param},
    activity_l2: {
        'l': float_param},
    activity_l1l2: {
        'l1': float_param,
        'l2': float_param}}
