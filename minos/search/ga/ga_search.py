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


def init_ga_env(experiment):
    if not isinstance(experiment, GaExperiment):
        raise Exception(
            'Unexpected experiment type %s'
            % str(experiment))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create(
        "Individual",
        experiment.individual_type,
        fitness=creator.FitnessMax)  # @UndefinedVariable
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


def evolve(population=None, population_size=50,
           population_age=0, generations=50, offspring_count=2, cx_prob=0.5,
           mutpb_prob=0.2, aliens_ratio=0.1, generation_logger=None):
    population = population or toolbox.population(n=population_size)
    for generation in range(population_age, generations):
        logging.info('Evolving generation %d' % generation)
        fit_invalid_individuals(population)
        population = list(sorted(
            population,
            key=lambda i: -i.fitness.values[0]))[:population_size]
        if generation_logger:
            generation_logger(generation, population)
        mates = tools.selBest(population, population_size)
        for ind1, ind2 in zip(mates[::2], mates[1::2]):
            if rand.random() < cx_prob:
                children = toolbox.mate(ind1, ind2, offspring_count)
                for child in children:
                    del child.fitness.values
                population += children
        for mutant in population:
            if rand.random() < mutpb_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        if aliens_ratio > 0:
            population += toolbox.population(n=int(population_size * aliens_ratio))
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


def search(experiment, population_size=50, generations=100,
           population_age=0, population=None, log_level='INFO', step_logger=None):
    init_ga_env(experiment)
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
        generation_logger=get_generation_logger(experiment, step_logger))


def _get_population_filename(output_dir, experiment_label):
    return path.join(output_dir, '%s.population' % experiment_label)


def get_generation_logger(experiment, step_logger=None):

    def _log(generation, population):
        log_generation_info(generation, population)
        if not step_logger:
            return
        step_logger(experiment, generation, population)

    return _log


def log_generation_info(generation, population):
    sorted_population = list(sorted(
        population,
        key=lambda i: -i.fitness.values[0]))
    population_scores = [
        individual.fitness.values[0]
        for individual in population]
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
