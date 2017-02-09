'''
Created on Feb 8, 2017

@author: julien
'''
from keras.engine.topology import Merge
from keras.layers.core import Dense, Dropout

from minos.architecture.parameters.constraint import constraints
from minos.architecture.parameters.parameter import ParameterConstraint,\
    boolean_param, float_param, to_namespace_id
from minos.architecture.parameters.regularizer import regularizers,\
    activity_regularizers


layers = {
    Dense: {
        'init': [
            'uniform',
            'lecun_uniform',
            'normal'
            'identity',
            'orthogonal',
            'zero',
            'one',
            'glorot_normal',
            'glorot_uniform',
            'he_normal',
            'he_uniform'],
        'activation': [
            None,
            'softmax',
            'softplus',
            'softsign',
            'relu',
            'tanh',
            'sigmoid',
            'hard_sigmoid',
            'linear'],
        'bias': boolean_param(),
        'W_regularizer': regularizers,
        'b_regularizer': regularizers,
        'activity_regularizer': activity_regularizers,
        'W_constraint': constraints,
        'b_constraint': constraints
    },
    Dropout: {
        'p': float_param()},
    Merge: {
        'mode': ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max']}
}


class LayerParameterConstraint(ParameterConstraint):

    def __init(self, layer, name, value, **kwargs):
        super().__init__(
            to_namespace_id('layers', layer),
            name,
            value,
            **kwargs)
