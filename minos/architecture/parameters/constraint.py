'''
Created on Feb 8, 2017

@author: julien
'''
from keras.constraints import maxnorm, nonneg, unitnorm

constraints = {
    maxnorm: {'m': ''},
    nonneg: {},
    unitnorm: {}}
