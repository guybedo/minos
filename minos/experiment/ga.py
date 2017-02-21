'''
Created on Feb 14, 2017

@author: julien
'''
from minos.experiment.experiment import Blueprint, Experiment, run_experiment
from minos.model.design import create_random_blueprint, mutate_blueprint,\
    mix_blueprints
from minos.search.ga.ga_search import search, GaExperiment, GaIndividual
from copy import deepcopy


class GaModelExperiment(Experiment, GaExperiment):

    def __init__(self, experiment):
        vars(self).update(vars(experiment))
        self.individual_type = GaBlueprint

    def evaluate(self, individuals):
        blueprints = [
            Blueprint(individual.layout, individual.training)
            for individual in individuals]
        return super().evaluate(blueprints)

    def random_individual(self):
        return GaBlueprint(individual=create_random_blueprint(self))

    def mutate_individual(self, individual):
        return GaBlueprint(
            individual=mutate_blueprint(
                individual,
                self.parameters,
                mutate_in_place=False))

    def mix_individuals(self, individual1, individual2):
        return GaBlueprint(
            individual=mix_blueprints(
                individual1,
                individual2,
                self.parameters))


class GaBlueprint(Blueprint, GaIndividual):

    def __init__(self, **kwargs):
        individual = kwargs.get('individual', None)
        if individual:
            vars(self).update(vars(individual))
        else:
            vars(self).update(kwargs)

    @classmethod
    def copy(cls, other_individual):
        return GaBlueprint(deepcopy(other_individual.blueprint))

    def clone(self):
        return GaBlueprint.copy(self)


def run_ga_search_experiment(experiment, population_size=50,
                             generations=100, resume=False):
    run_experiment(
        GaModelExperiment(experiment),
        search,
        resume=resume,
        population_size=population_size,
        generations=generations)
