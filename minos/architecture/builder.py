'''
Created on Feb 6, 2017

@author: julien
'''

def build_configuration(configuration):
    instance = dict(definition=configuration)
    instance['topology'] = build_topology(configuration.topology)
    instance['training'] = build_topology(configuration.training)
    return instance
    
def build_topology(topology_definition):
    instance = dict(definition=topology_definition)
    return instance
