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

    def mutate_individual(self, individual, mutate_in_place=False):
        return GaBlueprint(
            individual=mutate_blueprint(
                individual,
                self.parameters,
                p_mutate_layout=self.settings.ga['p_mutation'],
                p_mutate_param=self.settings.ga['p_mutation'],
                mutation_std=self.settings.ga['mutation_std'],
                mutate_in_place=mutate_in_place))

    def mix_individuals(self, individual1, individual2):
        mix = mix_blueprints(
            individual1,
            individual2,
            parameters=self.parameters,
            mutate_layout=self.settings.search['layout'],
            mutate_params=self.settings.search['parameters'],
            mutate_optimizer=self.settings.search['optimizer'],
            p_mutate_param=self.settings.ga['p_mutation'],
            mutation_std=self.settings.ga['mutation_std'])
        return GaBlueprint(individual=mix)


class GaBlueprint(Blueprint, GaIndividual):

    def __init__(self, **kwargs):
        individual = kwargs.get('individual', None)
        if individual:
            vars(self).update(vars(individual))
            score = getattr(individual, 'score', None)
            if score:
                self.fitness.values = score
        else:
            vars(self).update(kwargs)

    def to_blueprint(self):
        blueprint = Blueprint(**{
            k: v
            for k, v in vars(self).items()
            if k != 'fitness'})
        fitness = getattr(self, 'fitness', None)
        if fitness:
            blueprint.score = fitness.values
        return blueprint

    @classmethod
    def copy(cls, other_individual):
        return GaBlueprint(deepcopy(other_individual.blueprint))

    def clone(self):
        return GaBlueprint.copy(self)


def run_ga_search_experiment(experiment, resume=False, log_level='INFO'):
    run_experiment(
        GaModelExperiment(experiment),
        search,
        resume=resume,
        log_level=log_level)
