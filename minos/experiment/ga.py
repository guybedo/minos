'''
Created on Feb 14, 2017

@author: julien
'''
from minos.experiment.experiment import Blueprint
from minos.model.design import create_random_blueprint, mutate_blueprint,\
    mix_blueprints
from minos.search.ga.ga_experiment import GaExperiment, GaIndividual


class GaModelExperiment(GaExperiment):

    def __init__(self, experiment):
        super().__init__(experiment)

    def random_individual(self):
        return GaBlueprint(create_random_blueprint(self.experiment))


class GaBlueprint(GaIndividual):

    def __init__(self, blueprint):
        self.blueprint = blueprint

    def mutate(self):
        self.blueprint = mutate_blueprint(self.blueprint)
        return self

    @classmethod
    def mix(self, individual1, individual2):
        return mix_blueprints(
            individual1.blueprint,
            individual2.blueprint)

    @classmethod
    def copy(cls, other_individual):
        return GaBlueprint(Blueprint(**other_individual.blueprint))
