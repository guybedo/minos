'''
Created on Dec 5, 2016

@author: julien
'''
import json
import logging
import numpy
from os import path
from random import Random

from deap import creator, base, tools

from minos.model.build import ModelBuilder
from minos.tf_utils import cpu_device

toolbox = base.Toolbox()
rand = Random()


def random_individual(experiment, individual_type):
    return individual_type(individual=experiment.random_individual())


def make_individual(individual_type, original):
    return individual_type(individual=original)


def mutate(experiment, individual):
    experiment.mutate_individual(individual, mutate_in_place=True)


def mate(experiment, individual1, individual2, count=2):
    children = [
        experiment.mix_individuals(
            toolbox.clone(individual1),
            toolbox.clone(individual2))
        for _ in range(count)]
    return [
        toolbox.make_individual(child)
        for child in children]


def get_is_minimization_problem(experiment):
    return experiment.settings.ga['fitness_type'] == 'FitnessMin'


def get_fitness_weights(fitness_type):
    if fitness_type == 'FitnessMin':
        return (-1.0,)
    else:
        return (1.0,)


def init_ga_env(experiment):
    if not isinstance(experiment, GaExperiment):
        raise Exception(
            'Unexpected experiment type %s'
            % str(experiment))
    fitness_type = experiment.settings.ga['fitness_type']
    creator.create(fitness_type, base.Fitness, weights=get_fitness_weights(fitness_type))
    creator.create(
        "Individual",
        experiment.individual_type,
        fitness=getattr(creator, fitness_type))  # @UndefinedVariable
    toolbox.register(
        "individual",
        random_individual,
        experiment,
        creator.Individual)  # @UndefinedVariable
    toolbox.register(
        "make_individual",
        make_individual,
        creator.Individual)  # @UndefinedVariable
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutate, experiment)
    toolbox.register("mate", mate, experiment)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", experiment.evaluate)


def sort_population(population, population_size, minimize=False):
    return list(sorted(
        population,
        key=lambda i: -i.fitness.values[0] if not minimize else i.fitness.values[0]))[:population_size]


def evolve(population=None, population_size=50,
           population_age=0, generations=50, p_offspring=0.5,
           offsprings=2, p_aliens=0.25, aliens=0.1, generation_logger=None, fitness_type=False):
    population = population or toolbox.population(n=population_size)
    for generation in range(population_age, generations):
        logging.info('Evolving generation {} of {} with {}'.format(generation, generations, fitness_type))
        fit_invalid_individuals(population)
        population = sort_population(population, population_size, fitness_type == 'FitnessMin')
        if generation_logger:
            generation_logger(generation, population)
        mates = tools.selBest(population, population_size)
        for ind1, ind2 in zip(mates[::2], mates[1::2]):
            if rand.random() < p_offspring:
                children = toolbox.mate(ind1, ind2, offsprings)
                for child in children:
                    del child.fitness.values
                population += children
        if rand.random() < p_aliens:
            population += toolbox.population(n=int(population_size * aliens))
    return population


def fit_invalid_individuals(population):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    if len(invalid_ind) == 0:
        return
    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        if fit is None:
            fit = 0
        ind.fitness.values = fit


def search(experiment,
           population_age=0, population=None,
           log_level='INFO', step_logger=None):
    init_ga_env(experiment)
    population_size = experiment.settings.ga['population_size']
    generations = experiment.settings.ga['generations']
    print("HERE IN SEARCH")
    if population:
        population = [
            toolbox.make_individual(individual)
            for individual in population]
        if len(population) < population_size:
            population += toolbox.population(n=population_size - len(population))
    evolve(
        population=population,
        population_age=population_age,
        population_size=population_size,
        generations=generations,
        p_offspring=experiment.settings.ga['p_offspring'],
        offsprings=experiment.settings.ga['offsprings'],
        p_aliens=experiment.settings.ga['p_aliens'],
        aliens=experiment.settings.ga['aliens'],
        generation_logger=get_generation_logger(experiment, step_logger),
        fitness_type=experiment.settings.ga['fitness_type'])


def _get_population_filename(output_dir, experiment_label):
    return path.join(output_dir, '%s.population' % experiment_label)


def get_generation_logger(experiment, step_logger=None):
    def _log(generation, population):
        log_generation_info(experiment, generation, population)
        if not step_logger:
            return
        step_logger(experiment, generation, population)

    return _log


def log_generation_info(experiment, generation, population):
    minimize = get_is_minimization_problem(experiment)
    sorted_population = list(sorted(
        population,
        key=lambda i: -i.fitness.values[0] if not minimize else i.fitness.values[0]))
    population_scores = [
        individual.fitness.values[0]
        for individual in population]

    model = ModelBuilder().build(
        population[0],
        cpu_device(),
        compile_model=True)

    generation_info = [
        {'generation': generation},
        {'average': numpy.mean(population_scores)},
        {'best_scores': [
            best.fitness.values[0]
            for best in sorted_population[:3]]}]
    logging.info(json.dumps(generation_info))


class GaExperiment(object):
    def __init__(self):
        pass

    def random_individual(self):
        pass


class GaIndividual(object):
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def mutate(self):
        pass

    @classmethod
    def mix(cls, individual1, individual2):
        pass

    @classmethod
    def copy(cls, other_individual):
        pass
