'''
Created on Feb 14, 2017

@author: julien
'''

from minos.model.parameter import int_param, string_param, boolean_param,\
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
            'input_size': float_param(default=1.)}
    },
    'layers': {
        'Dense': {
            'init': string_param(
                ['uniform',
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
                default='glorot_normal'),
            'activation': string_param(
                [None,
                 'softmax',
                 'softplus',
                 'softsign',
                 'relu',
                 'tanh',
                 'sigmoid',
                 'hard_sigmoid',
                 'linear'],
                default='relu'),
            'bias': boolean_param(default=True),
            'W_regularizer': {
                'l1': float_param(optional=True),
                'l2': float_param(optional=True)},
            'b_regularizer': {
                'l1': float_param(optional=True),
                'l2': float_param(optional=True)},
            'activity_regularizer': {
                'l1': float_param(optional=True),
                'l2': float_param(optional=True)},
            'W_constraint': string_param(
                [None, 'maxnorm', 'nonneg', 'unitnorm'],
                default=None),
            'b_constraint': string_param(
                [None, 'maxnorm', 'nonneg', 'unitnorm'],
                default=None)
        },
        'Dropout': {
            'p': float_param(default=0.75)},
        'Merge': {
            'mode': string_param(
                ['sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'],
                default='concat')},
        'BatchNormalization': {
            'epsilon': float_param(default=0.001),
            'mode': int_param(values=[0, 1, 2], default=0),
            'axis': int_param(default=1),
            'momentum': float_param(default=0.99),
            'beta_init': string_param(
                ['uniform',
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
                default='glorot_normal'),
            'gamma_init': string_param(
                ['uniform',
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
                default='glorot_normal'),
            'gamma_regularizer': {
                'l1': float_param(optional=True),
                'l2': float_param(optional=True)},
            'beta_regularizer': {
                'l1': float_param(optional=True),
                'l2': float_param(optional=True)}
        }
    },
    'optimizers': {
        'SGD': {
            'lr': float_param(default=1e-3),
            'momentum': float_param(),
            'decay': float_param(),
            'nesterov': boolean_param()},
        'RMSprop': {
            'lr': float_param(default=1e-3),
            'rho': float_param(default=0.9),
            'epsilon': float_param(default=1e-08),
            'decay': float_param()},
        'Adagrad': {
            'lr': float_param(default=1e-3),
            'epsilon': float_param(default=1e-08),
            'decay': float_param()},
        'Adadelta': {
            'lr': float_param(default=1e-3),
            'rho': float_param(default=0.9),
            'epsilon': float_param(default=1e-08),
            'decay': float_param()},
        'Adam': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'decay': float_param()},
        'Adamax': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'decay': float_param()},
        'Nadam': {
            'lr': float_param(default=1e-3),
            'beta_1': float_param(default=0.9),
            'beta_2': float_param(default=0.999),
            'epsilon': float_param(default=1e-08),
            'schedule_decay': float_param()}},
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
