'''
Created on Feb 7, 2017

@author: julien
'''
import unittest

from minos.architecture.parameters.layer import layers
from minos.architecture.parameters.param_space import build_parameter_space


class ParameterSpaceTest(unittest.TestCase):

    def test_parameter_space(self):
        param_space = build_parameter_space()
        for layer in layers.keys():
            for name, value in layers[layer].items():
                self.assertIsNotNone(
                    param_space.get_layer_parameter(layer, name),
                    'Parameter %s should exist' % name)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
