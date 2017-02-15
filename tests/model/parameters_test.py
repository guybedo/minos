'''
Created on Feb 7, 2017

@author: julien
'''
import unittest

from minos.experiment.experiment import ExperimentParameters
from minos.model.parameter import random_param_value, int_param, float_param
from minos.model.parameters import reference_parameters


class ParametersTest(unittest.TestCase):

    def test_parameters(self):
        experiment_parameters = ExperimentParameters()
        for layer in reference_parameters['layers'].keys():
            for name, _value in reference_parameters['layers'][layer].items():
                self.assertIsNotNone(
                    experiment_parameters.get_layer_parameter(layer, name),
                    'Parameter %s should exist' % name)

    def test_random_value(self):
        param = int_param(values=list(range(10)))
        val = random_param_value(param)
        self.assertTrue(
            isinstance(val, int),
            'Should be an int')
        self.assertTrue(
            val in param.values,
            'Value should be in predefined values')

        param = float_param(values=[i * 0.1 for i in range(10)])
        val = random_param_value(param)
        self.assertTrue(
            isinstance(val, float),
            'Should be a float')
        self.assertTrue(
            val in param.values,
            'Value should be in predefined values')

        param = float_param(lo=.5, hi=.7)
        for _ in range(100):
            val = random_param_value(param)
            self.assertTrue(
                isinstance(val, float),
                'Should be a float')
            self.assertTrue(
                val <= param.hi and val >= param.lo,
                'Value should be in range')

        param = {
            'a': float_param(optional=False),
            'b': float_param(optional=False)}
        for _ in range(10):
            val = random_param_value(param)
            self.assertTrue(
                isinstance(val, dict),
                'Should be a dict')
            self.assertEqual(
                len(param), len(val),
                'Should respect non optional setting')

        param = {
            'a': float_param(optional=True),
            'b': float_param(optional=True)}
        for _ in range(10):
            val = random_param_value(param)
            self.assertTrue(
                isinstance(val, dict),
                'Should be a dict')
            self.assertTrue(
                len(val) >= 0 and len(val) <= len(param),
                'Should respect non optional setting')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
