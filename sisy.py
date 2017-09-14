from copy import deepcopy

from keras.datasets import boston_housing
from keras.layers import Dense

from minos.experiment.experiment import Experiment, ExperimentSettings
from minos.experiment.ga import run_ga_search_experiment
from minos.experiment.training import Training, EpochStoppingCondition
from minos.model.model import Layout, Objective, Optimizer, Metric
from minos.model.parameter import int_param
from minos.model.parameters import register_custom_layer, reference_parameters
from minos.train.utils import SimpleBatchIterator, GpuEnvironment
from minos.experiment.experiment import ExperimentParameters

def default(x,d):
    if x:
        return x
    return d

def run_sisy_experiment(sisy_layout : list,
                        experiment_label : str,
                        XYTrainTuple : tuple,
                        XYTestTuple : tuple,
                        generations=10,
                        batch_size=1,
                        autoloop=True,
                        population_size=25,
                        epochs=50,
                        devices =['/gpu:0','/gpu:1'],
                        n_jobs = 1,
                        optimizer = 'sgd',
                        loss = 'categorical_crossentropy',
                        metric='acc',
                        offspring = 1,
                        mutation = 1,
                        fitness_type='FitnessMax',
                        shuffle=True):


    if len(sisy_layout) < 2:
        print("Sisy Layout must be at least size 2, an output, middle, and output layer")
        return;

    if len(XYTrainTuple) != 2:
        print("XYTrainTuple must be a tuple of length 2, (X_train,y_train) ")
        return;
    if len(XYTestTuple) != 2:
        print("XYTrainTuple must be a tuple of length 2, (X_train,y_train) ")
        return;


    X_train = XYTrainTuple[0]
    y_train = XYTrainTuple[1]

    X_test = XYTestTuple[0]
    y_test  = XYTestTuple[1]

    input =sisy_layout[0]
    output = sisy_layout[-1]


    if 'activation' not in output[1]:
        print("You must specify the parameter 'activation' for the Output layer");
        return;
    if 'units' not in output[1]:
            print("You must specify the parameter 'units' for the Output layer");
            return;
    if 'units' not in input[1]:
            print("You must specify the parameter 'units' for the Input layer");
            return;

    output_activation = default(output[1]['activation'],'categorical_crossentropy')
    output_initializer = 'normal'
    if 'kernel_initializer' in output[1]:
        output_initializer = output[1]['kernel_initializer']

    output_size = output[1]['units']
    input_size = default(input[1]['units'],X_train.shape[1])


    batch_iterator = SimpleBatchIterator(X_train, y_train, batch_size=batch_size, autoloop=autoloop, preload=True, simple=not shuffle)
    test_batch_iterator = SimpleBatchIterator(X_test, y_test, batch_size=batch_size, autoloop=autoloop, preload=True, simple=not shuffle)

    # our training , MSE for the loss and metric, stopping condition of 5 since our epochs are only 10

    training = Training(
       Objective(loss),
       Optimizer(optimizer=optimizer),
       Metric(metric),
       EpochStoppingCondition(epochs),
       1)




    blocks = []
    units_list = []
    for i,e in enumerate(sisy_layout[1:-1]):
        block = ()
        if e[0] in ['Dense']:
            key = f'Dense{i}'
            print("Registering {}".format(key))
            register_custom_layer(
                   key,
                   Dense,
                   deepcopy(reference_parameters['layers']['Dense']),
                   True)
            layer = deepcopy(e[1])
            units_list.append( layer['units'] )
            del layer['units']
            block  =  (key, layer)
        else:
            block = e
        blocks.append(block )


    print(blocks)
    layout = Layout(
       input_size,  # Input size, 13 features I think
       output_size ,  # Output size, we want just the price
       output_activation=output_activation,  # linear activation since its continous number
       output_initializer=output_initializer,
       # Our template, just one block with two dense layers
       block=blocks
    )


    experiment_parameters = ExperimentParameters(use_default_values=True)
    experiment_settings = ExperimentSettings()

    experiment_parameters.layout_parameter('rows', 1)
    experiment_parameters.layout_parameter('blocks', 1)
    experiment_parameters.layout_parameter('layers', 1)


    for i,units in enumerate(units_list):
        key = f'Dense{i}.units'
        if type(units) == int:
            experiment_parameters.layer_parameter(key, units)
        else:
            experiment_parameters.layer_parameter(key, int_param(units[0], units[-1]))



    experiment_settings.ga['population_size'] = population_size
    experiment_settings.ga['generations'] = generations
    experiment_settings.ga['p_offspring'] = offspring
    experiment_settings.ga['p_mutation'] = mutation

    # TO specify minimizing the loss , lets use FitnessMin for a evolution criteria
    experiment_settings.ga['fitness_type'] = fitness_type

    experiment = Experiment(
       experiment_label,
       layout=layout,
       training=training,
       batch_iterator=batch_iterator,
       test_batch_iterator=test_batch_iterator,
       environment=GpuEnvironment(devices=devices, n_jobs=n_jobs),
       parameters=experiment_parameters,
       settings=experiment_settings
    )

    run_ga_search_experiment(
       experiment,
       resume=False,
       log_level='DEBUG')

