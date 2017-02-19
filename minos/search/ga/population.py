'''
Created on Feb 11, 2017

@author: julien
'''
import json
import logging
from os.path import isfile
import pickle

from numpy import mean


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
    generation_best = sorted(
        population,
        key=lambda i: -i.fitness.values[0])[:3]
    population_scores = [
        individual.fitness.values[0]
        for individual in population]
    generation_info = list()
    generation_info.append({'generation': generation_info})
    generation_info.append({'average': mean(population_scores)})
    generation_info.append({'best_scores': [
        best.fitness.values[0]
        for best in generation_best]})
    generation_info.append({'best_individuals': [
        dict(score=best.fitness.values[0], **vars(best))
        for best in generation_best]})
    logging.info(json.dumps(generation_info))
