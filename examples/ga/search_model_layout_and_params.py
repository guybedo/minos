'''
Created on Feb 6, 2017

@author: julien
'''
from minos.experiment.blueprint import Blueprint
from minos.model.parameters.layout import Layout
from minos.model.parameters.metric import Metric
from minos.model.parameters.model import ModelParameters
from minos.model.parameters.objective import Objective
from minos.model.parameters.optimizer import Optimizer,\
    OptimizerParameters
from minos.model.parameters.training import Training,\
    MetricDecreaseStoppingCondition


def search():
    input_size = 10
    output_size = 2
    train, test = None, None
    batch_size = 1000
    epoch_steps = int(len(train) / batch_size)

    layout = Layout(
        layout_parameters=LayoutParameters(),
        model_parameters=ModelParameters())
    training = Training(
        Objective('mean_squared_error'),
        Optimizer(parameters=OptimizerParameters()),
        Metric('binary_accuracy'),
        MetricDecreaseStoppingCondition(
            min_step_count=epoch_steps,
            max_step_count=25 * epoch_steps,
            flat_step_count=3,
            measurement_interval=epoch_steps))
    return Blueprint(layout, training)
