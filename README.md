# Minos

Search for neural networks architecture & hyper parameters with genetic algorithms.
It is built on top of Keras+Tensorflow to build/train/evaluate the models, and uses DEAP for the genetic algorithms.

## Getting Started

You need to have tensorflow installed, see [tensorflow linux](requirements-tensorflow-linux.txt) or [tensorflow mac](requirements-tensorflow-mac.txt)

Install minos:
```
pip install pyminos==0.5.1
```

To run an experiment and search hyper parameters and/or architecture for a model and dataset, you can define a simple layout
with the input_size, output_size and output_activation of your model

```python
from minos.model.model import Layout
layout = Layout(
    input_size=1000,
    output_size=25,
    output_activation='softmax')
```

Then you define the parameters of the training. If you specify only the name of the optimizer to use, and no parameters, random parameters will be tested during the experiment, hopefully converging  to optimal parameters.
You can choose to stop the training after a fixed number of epochs, or when the accuracy of the model evaluated stops increasing.

```python
from minos.model.model import Objective, Optimizer, Metric
from minos.experiment.training import Training, EpochStoppingCondition
training = Training(
    objective=Objective('categorical_crossentropy'),
    optimizer=Optimizer(optimizer='Adam'),
    metric=Metric('categorical_accuracy'),
    stopping=EpochStoppingCondition(10),
    batch_size=50)
```

Now you need to define which parameters will be randomly tested. 
An ExperimentParameters contains all the parameters that can be tested. It can be initialized with the default values for each parameter so that you only redefine the parameters you want to test, specifying intervals or list of values for example

```python
from minos.experiment.experiment import ExperimentParameters
experiment_parameters = ExperimentParameters(use_default_values=True)
```

You can then specify the search space for each parameter you want to test.
For example, to test architectures with 1 row, 1 block per row, and up to 5 layers per block: 

```python
from minos.model.parameter import int_param
experiment_parameters.layout_parameter('rows', 1)
experiment_parameters.layout_parameter('blocks', 1)
experiment_parameters.layout_parameter('layers', int_param(1, 5))
```

If you want to test layers with size between 10 and 500 units:
```python
experiment_parameters.layer_parameter('Dense.output_dim', int_param(10, 500))
```

You can find all the parameters and their default values here in [parameters] (minos/model/parameters.py)

Now you need to specify the experiment environment. 
You can choose to run the experiment on CPU or GPU devices, and specify how many jobs are to be run on each device. To run on CPU, just use CpuEnvironment instead of GpuEnvironment.
You can define the directory where the experiment logs and data are saved. If no directory defined, it will create a directory named 'minos' in the user's home.

```python
from minos.train.utils import GpuEnvironment
environment=GpuEnvironment(
    ['/gpu:0', '/gpu:1'], 
    n_jobs=[2, 5],
    data_dir='/data/minos/experiments')
```

The Experiment is then created with all the information necessary and the training and validation data.
Training and validation data are provided as batch iterators that generate (X,y) tuples.
You can use SimpleBatchIterator to create a batch iterator from (X, y) arrays. The iterators need to be able to loop over the data when they reach the end, so you need to set the parameter autoloop=True.

```python
from minos.train.utils import SimpleBatchIterator
batch_iterator = SimpleBatchIterator(X, y, batch_size=50, autoloop=True)
test_batch_iterator = SimpleBatchIterator(test_X, test_y, batch_size=50, autoloop=True)
from minos.experiment.experiment import Experiment
experiment = Experiment(
    experiment_label='test__reuters_experiment',
    layout=layout,
    training=training,
    batch_iterator=batch_iterator,
    test_batch_iterator=test_batch_iterator,
    environment=environment,
    parameters=experiment_parameters)
```

Then you specify the population size and number of generations and start the experiment.
Logs and data will be saved in the directory you specified.

```python
from minos.experiment.ga import run_ga_search_experiment
run_ga_search_experiment(
    experiment, 
    population_size=100, 
    generations=100,
    log_level='DEBUG')
```

Logs and data will be saved in the specified directory, or ~/minos if no directory specified.
This is what the logs should look like 
```terminal
2017-02-21 07:25:26 [INFO] root: Evolving generation 0
2017-02-21 07:25:26 [DEBUG] root: Training 100 models
2017-02-21 07:27:29 [DEBUG] root: Blueprint 0: score 0.438252 after 17 epochs
2017-02-21 07:27:31 [DEBUG] root: Blueprint 1: score 0.326195 after 20 epochs
2017-02-21 07:28:13 [DEBUG] root: Blueprint 3: score 0.496040 after 22 epochs
2017-02-21 07:29:26 [DEBUG] root: Blueprint 4: score 0.835436 after 24 epochs
2017-02-21 07:30:18 [DEBUG] root: Blueprint 5: score 0.261954 after 21 epochs
2017-02-21 07:31:02 [DEBUG] root: Blueprint 2: score 0.096509 after 51 epochs
2017-02-21 07:35:14 [DEBUG] root: Blueprint 7: score 0.370490 after 36 epochs
2017-02-21 07:38:12 [DEBUG] root: Blueprint 6: score 0.537401 after 104 epochs
2017-02-21 07:40:25 [DEBUG] root: Blueprint 8: score 0.176298 after 57 epochs
2017-02-21 07:41:08 [DEBUG] root: Blueprint 11: score 0.063068 after 24 epochs
2017-02-21 07:45:55 [DEBUG] root: Blueprint 10: score 0.022587 after 65 epoch
2017-02-21 10:02:29 [INFO] root: [{"generation": 0}, {"average": 0.36195365556387343}, {"best_scores": [0.842769172996606, 0.8392491032735243, 0.8354356464279401]}]
```

You can stop the experiment and resume later by setting the 'resume' parameter to True. It will restart at the last epoch saved. 
```python
run_ga_search_experiment(
    experiment, 
    population_size=100, 
    generations=100,
    resume=True)
```

Once you are done, you can load the best blueprint produced at a specific step.
```python
from minos.experiment.experiment import load_experiment_best_blueprint
blueprint = load_experiment_best_blueprint(
    experiment_label=experiment.label,
    step=generations - 1,
    environment=CpuEnvironment(n_jobs=2, data_dir=tmp_dir))
```    
    
And then build/train/evaluate the model using the Keras API:
```python
from minos.model.build import ModelBuilder
from minos.train.utils import cpu_device
model = ModelBuilder().build(
    blueprint,
    cpu_device())
model.fit_generator(
    generator=batch_iterator,
    samples_per_epoch=batch_iterator.samples_per_epoch,
    nb_epoch=5,
    validation_data=test_batch_iterator,
    nb_val_samples=test_batch_iterator.sample_count)
score = model.evaluate_generator(
    test_batch_iterator,
    val_samples=test_batch_iterator.sample_count)
```

## Limitations
The current version only works with 1D data, so no RNN, LSTM, Convolutions for now...


## Concepts
To search for hyper parameters and/or layouts, we create an experiment.
We define the parameters of the experiment and the dataset, then we run the experiment.
An experiment uses a genetic algorithm to search the parameters defined.
It consists in generating a population, and evolving the population for a specified number
of generations.
It starts by generating a random population of blueprints from the experiment parameters.
Each blueprint, or individual, randomly generated, is actually a definition that can be used
to build a Keras model.
At each generation, the blueprints can be mixed and/or mutated, and are then evaluated.
Evaluating a blueprint consists in building, training and evaluating the Keras model if defines.
The best blueprints are selected for the next generation

To create an experiment you need to define:
- the layout:
    input_size, output_size, output_activtion of the network.
      You can also specify the architecture and layers if you want to search parameters
      for a fixed architecture.
      If you don't specify any layers, random combinations will be tested.
- the experiment parameters:
    these are all the parameters that will be randomly tested
      You can decide to test every possible combination, or fix the value of some parameters and
      let the experiment randomly test others
- the training:
    objective(=loss), metric, stopping condition and optimizer.
    These training parameters are used to evaluate the models randomly generated.
      Note that you can either fully specify the optimizer (type+parameters) or specify only
      a type of optimizer and let the experiment test random parameters

## Terminology

    Experiment:
        Defines all the parameters related to the search:
            - the layout,
            - the layer parameters
            - the training parameters
    Layout:
        A layout defines the architecture of a network. A layout is vertical stack of rows.
    Row:
        A row is an horizontal stack of independant blocks. Each block can be connected to or more blocks
        from the row below.
    Block:
        A block is vertical stack of layers. The output of each layer in the block is the input of the
        layer immediately above
    Layer:
        A Keras layer : Dense, Dropout, ...

    ExperimentParameters:
        Defines all the parameters that can be tested. This can be layer parameters such as the dropout value,
        the regularization type and value, etc... this can be the parameters of the optimizer...
        You can :
            - initialize the ExperimentParameters with the default values for each parameters.
              In that case you then need to override the parameters you want to search and specify
              the intervals or collections of values to be randomly tested
            - initialize the ExperimentParameters without default values.
              In that case all the parameters will be randomly tested
        The reference parameter intervals and default values can be found in minos.model.parameters

    Training:
        Defines the training parameters used to evaluate the randomly generated models.
        You specify the objective(loss), the metric, the stopping condition and the optimizer.
        The hyper parameters for the optimizers can also be randomly tested


    Blueprint:
        Blueprints are generated randomly from the experiment parameters you specify.
        A blueprint is the definition that is used to build and train/evaluate a Keras model.
        During the experiment random blueprints are generated, mixed, mutated and evaluated
        by training and evaluating the Keras model they define

    Model:
        A Keras model built using a blueprint

## Documentation
For now there is no documentation. Best thing to do is to have a look at the examples in https://github.com/guybedo/minos/tree/develop/examples.
This is quite straightforward to use, the examples should be enough to start trying things and running experiments.
