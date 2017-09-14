'''
Created on Feb 14, 2017

@author: julien
'''

from minos.model.parameter import int_param, string_param, boolean_param, \
    float_param

reference_parameters = {
    'layout': {
        'rows': int_param(lo=1, hi=3, default=1),
        'blocks': int_param(lo=1, hi=5, default=1),
        'layers': int_param(lo=1, hi=5, default=1),
        'block': {
            'input_type': string_param([
                'concat',
                'random+concat',
                'concat+random'], default='concat'),
            'input_size': float_param(default=1.)},
        'layer': {
            'type': string_param(['Dense', 'Dropout', 'BatchNormalization']),
            'stackable': string_param([])}
    },
    'layers': {
        'Embedding': {
            'output_dim': int_param(1, 10000, default=1),
            'input_dim': int_param(1, 10000, default=1),
        },
        'Dense': {
            'units': int_param(1, 1000, default=100),
            'kernel_initializer': string_param(
                ['uniform',
                 'lecun_uniform',
                 'normal',
                 # 'identity',
                 # 'orthogonal',
                 'zero',
                 'one',
                 'glorot_normal',
                 'glorot_uniform',
                 'he_normal',
                 'he_uniform'],
                default='glorot_uniform'),
            'activation': string_param(
                ['softmax',
                 'softplus',
                 'softsign',
                 'relu',
                 'tanh',
                 'sigmoid',
                 'hard_sigmoid',
                 'linear'],
                default='relu'),
            'use_bias': boolean_param(default=True),
            'kernel_regularizer': {
                'l1': float_param(optional=True, default=None),
                'l2': float_param(optional=True, default=None)
            },
            'bias_regularizer': {
                'l1': float_param(optional=True, default=None),
                'l2': float_param(optional=True, default=None)
            },
            'activity_regularizer': {
                'l1': float_param(optional=True, default=None),
                'l2': float_param(optional=True, default=None)
            },
            'kernel_constraint': string_param(
                ['maxnorm', 'nonneg', 'unitnorm'],
                default=None,
                optional=True),
            'bias_constraint': string_param(
                ['maxnorm', 'nonneg', 'unitnorm'],
                default=None,
                optional=True)
        },
        'LSTM': {
            'units': int_param(1, 1000, default=100),
            'dropout': float_param(default=0.0),
            'recurrent_dropout': float_param(default=0.0),
            'kernel_initializer': string_param(
                ['uniform',
                 'lecun_uniform',
                 'normal',
                 # 'identity',
                 # 'orthogonal',
                 'zero',
                 'one',
                 'glorot_normal',
                 'glorot_uniform',
                 'he_normal',
                 'he_uniform'],
                default='glorot_uniform'),
            'recurrent_initializer': string_param(
                ['uniform',
                 'lecun_uniform',
                 'normal',
                 'identity',
                 'orthogonal',
                 'zero',
                 'one',
                 'glorot_normal',
                 'glorot_uniform',
                 'he_normal',
                 'he_uniform'],
                default='orthogonal'),
            'recurrent_activation': string_param(
                ['softmax',
                 'softplus',
                 'softsign',
                 'relu',
                 'tanh',
                 'sigmoid',
                 'hard_sigmoid',
                 'linear'],
                default='hard_sigmoid'),
            'activation': string_param(
                ['softmax',
                 'softplus',
                 'softsign',
                 'relu',
                 'tanh',
                 'sigmoid',
                 'hard_sigmoid',
                 'linear'],
                default='tanh'),
            'use_bias': boolean_param(default=True),
            'kernel_constraint': string_param(
                ['maxnorm', 'nonneg', 'unitnorm'],
                default=None,
                optional=True),
            'bias_constraint': string_param(
                ['maxnorm', 'nonneg', 'unitnorm'],
                default=None,
                optional=True)
        },
        'Dropout': {
            'rate': float_param(default=0.75)},
        'Merge': {
            'mode': string_param(
                ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'],
                default='concat',
                mutable=False)},
        'BatchNormalization': {
            'epsilon': float_param(default=0.001),
            'momentum': float_param(default=0.99),
            'beta_initializer': string_param(
                ['uniform',
                 'lecun_uniform',
                 'normal',
                 # 'identity',
                 # 'orthogonal',
                 'zero',
                 'one',
                 'glorot_normal',
                 'glorot_uniform',
                 'he_normal',
                 'he_uniform'],
                default='glorot_normal'),
            'gamma_initializer': string_param(
                ['uniform',
                 'lecun_uniform',
                 'normal',
                 # 'identity',
                 # 'orthogonal',
                 'zero',
                 'one',
                 'glorot_normal',
                 'glorot_uniform',
                 'he_normal',
                 'he_uniform'],
                default='glorot_normal'),
            'gamma_regularizer': {
                'l1': float_param(optional=True, default=None),
                'l2': float_param(optional=True, default=None)},
            'beta_regularizer': {
                'l1': float_param(optional=True, default=None),
                'l2': float_param(optional=True, default=None)}
        }
    },
    'optimizers': {
        'sgd': {
            'lr': float_param(default=1e-3),
            'momentum': float_param(default=0.0),
            'decay': float_param(default=0.0),
            'nesterov': boolean_param(default=False)},
        'rmsprop': {
            'lr': float_param(default=1e-3),
            'rho': float_param(default=0.9),
            'epsilon': float_param(default=1e-08),
            'decay': float_param(default=0.0)},
        'adagrad': {
            'lr': float_param(default=1e-3),
            'epsilon': float_param(default=1e-08),
            'decay': float_param(default=0.0)},
        'adadelta': {
            'lr': float_param(default=1e-3),
            'rho': float_param(default=0.9),
            'epsilon': float_param(default=1e-08),
            'decay': float_param(default=0.0)},
        'adam': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'decay': float_param(default=0.0)},
        'adamax': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'decay': float_param(default=0.0)},
        'nadam': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'schedule_decay': float_param(default=4e-3)}},
    'metric': string_param([
        'binary_accuracy',
        'categorical_accuracy',
        'sparse_categorical_accuracy',
        'top_k_categorical_accuracy',
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'hinge',
        'squared_hinge',
        'categorical_crossentropy',
        'sparse_categorical_crossentropy',
        'binary_crossentropy',
        'kullback_leibler_divergence',
        'poisson',
        'cosine_proximity',
        'matthews_correlation',
        'precision',
        'fbeta_score',
        'fmeasure'
    ]),
    'objective': string_param([
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'squared_hinge',
        'hinge',
        'binary_crossentropy',
        'categorical_crossentropy',
        'sparse_categorical_crossentropy',
        'kullback_leibler_divergence',
        'poisson',
        'cosine_proximity'
    ])
}

custom_layers = dict()
custom_activations = dict()


def is_custom_layer(name):
    return name in custom_layers


def is_custom_activation(name):
    return name in custom_activations


def get_custom_layers():
    return dict(custom_layers)


def get_custom_layer(name):
    return custom_layers.get(name, None)


def get_custom_activations():
    return dict(custom_activations)


def get_custom_activation(name):
    return custom_activations.get(name, None)


def register_custom_layer(name, layer, params=None, stackable=False):
    custom_layers[name] = (layer, params, stackable)


def register_custom_activation(name, activation):
    custom_activations[name] = activation
