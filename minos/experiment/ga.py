'''
Created on Feb 14, 2017

@author: julien
'''
from minos.experiment.experiment import Blueprint, Experiment
from minos.model.design import create_random_blueprint, mutate_blueprint,\
    mix_blueprints
from minos.search.ga.ga_search import search, GaExperiment, GaIndividual
from copy import deepcopy


class GaModelExperiment(Experiment, GaExperiment):

    def __init__(self, experiment):
        vars(self).update(vars(experiment))
        self.individual_type = GaBlueprint

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


def run_ga_search_experiment(experiment):
    search(GaModelExperiment(experiment))
