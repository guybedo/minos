'''
Created on Dec 5, 2016

@author: julien
'''
from genericpath import isfile
from os import path
from random import Random

from deap import creator, base, tools

from minos.search.ga.population import save_population,\
    load_population, log_generation_info
from minos.utils import setup_logging


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
           mutpb_prob=0.2, aliens_ratio=0.1, population_filename=None):
    population = population or toolbox.population(n=population_size)
    for generation in range(generations):
        fit_invalid_individuals(population)
        population = list(sorted(
            population,
            key=lambda i: -i.fitness.values[0]))[:population_size]
        save_population(population, population_filename)
        log_generation_info(generation, population)
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


def _get_population_filename(output_dir, experiment_label):
    return path.join(output_dir, '%s.population' % experiment_label)


def _get_log_filename(output_dir, experiment_label):
    return path.join(output_dir, '%s.log' % experiment_label)


def search(experiment, population_size=50,
           generations=100, resume=False, log_level='INFO'):
    setup_logging(
        _get_log_filename(
            experiment.environment.data_dir,
            experiment.label),
        log_level,
        resume=resume)
    init_ga_env(experiment)
    population = None
    population_filename = _get_population_filename(
        experiment.environment.data_dir,
        experiment.label)
    if resume and isfile(population_filename):
        population = load_population(population_filename)
        if len(population) < population_size:
            population += toolbox.population(n=population_size - len(population))
    evolve(
        population=population,
        population_size=population_size,
        generations=generations,
        population_filename=population_filename)


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
