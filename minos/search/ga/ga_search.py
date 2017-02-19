'''
Created on Dec 5, 2016

@author: julien
'''
from genericpath import isfile
import json
import logging
from os import path
import pickle
from random import Random
from statistics import mean

from deap import creator, base, tools


toolbox = base.Toolbox()
rand = Random()


def random_individual(experiment, individual_type):
    return individual_type(individual=experiment.random_individual())


def make_individual(individual_type, original):
    return individual_type(individual=original)


def mutate(experiment, individual):
    return toolbox.make_individual(experiment.mutate_individual(individual))


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
           generations=50, offspring_count=2, cx_prob=0.5,
           mutpb_prob=0.2, aliens_ratio=0.1, generation_logger=None):
    population = population or toolbox.population(n=population_size)
    for generation in range(generations):
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
        mutants = [
            mutant
            for mutant in population
            if rand.random() < mutpb_prob]
        population += [
            toolbox.mutate(mutant)
            for mutant in mutants]
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
           resume=False, log_level='INFO', step_logger=None):
    init_ga_env(experiment)
    population = None
    if resume:
        population_filename = _get_population_filename(
            experiment.get_experiment_data_dir(),
            experiment.label)
        if not isfile(population_filename):
            raise Exception('Previous population file not found, resume impossible')
        population = load_population(population_filename)
        if len(population) < population_size:
            population += toolbox.population(n=population_size - len(population))
    evolve(
        population=population,
        population_size=population_size,
        generations=generations,
        generation_logger=build_generation_logger(experiment, step_logger))


def _get_population_filename(output_dir, experiment_label):
    return path.join(output_dir, '%s.population' % experiment_label)


def build_generation_logger(experiment, step_logger=None):

    population_filename = _get_population_filename(
        experiment.get_experiment_data_dir(),
        experiment.label)

    def _log(generation, population):
        save_population(population, population_filename)
        log_generation_info(
            generation,
            population)
        if step_logger:
            step_logger(experiment, generation, population)

    return _log


def load_population(population_filename):
    logging.info('Loading population %s' % population_filename)
    if not isfile(population_filename):
        logging.info('Population file %s not found!' % population_filename)
        return None
    with open(population_filename, 'rb') as population_file:
        return pickle.load(population_file)


def save_population(population, population_filename):
    if not population_filename:
        return
    with open(population_filename, 'wb') as population_file:
        pickle.dump(population, population_file, -1)


def log_generation_info(generation, population):
    sorted_population = list(sorted(
        population,
        key=lambda i: -i.fitness.values[0]))
    population_scores = [
        individual.fitness.values[0]
        for individual in population]
    generation_info = [
        {'generation': generation},
        {'average': mean(population_scores)},
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
