# Minos

Search for neural networks architecture & hyper parameters with genetic algorithms.
It is built on top of Keras+Tensorflow to build/traing/evaluate the models, and uses DEAP for the genetic algorithms.

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
This isn't complex to use, the examples should be enough to start trying things and running experiments.
