'''
Created on Feb 6, 2017

@author: julien
'''


class Element(object):

    def __init__(self, **kwargs):
        self.__dict__.update({
            k: v
            for k, v in kwargs.items()
            if k in self.properties})


class Guidelines(Element):

    properties = [
        'topology',
        'parameter_space',
        'parameter_constraints', 
        'training']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def create_blueprint(self):
        pass


class Topology(Element):

    properties = ['input_size', 'output_size', 'brick', 'bricks']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def default_topology(input_size, output_size):
    return Topology(
        input_size=input_size,
        output_size=output_size,
        bricks=1)


class Brick(Element):

    properties = ['block', 'blocks']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Block(Element):

    properties = [
        'input_size',
        'output_size',
        'layers',
        'layer_size',
        'activation',
        'batch_normalization',
        'dropout']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Activation(object):
    types = ['sigmoid', 'tanh', 'relu']

    def __init__(self):
        pass


class Training(Element):

    properties = ['loss', 'optimizer', 'stopping']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Stopping(Element):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FixedStopping(Element):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AccuracyStopping(Element):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Loss(Element):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Optimizer(Element):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
