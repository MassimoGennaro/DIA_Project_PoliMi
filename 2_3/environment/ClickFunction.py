import numpy as np


def disaggregate_function(x, user, phase):
    if user == 'FaY' and phase == 'M':
        return (1-np.exp(-x)) * 100
    if user == 'FaY' and phase == 'E':
        return (1-np.exp(-x)) * 150
    if user == 'FaY' and phase == 'W':
        return (1-np.exp(-x)) * 300
    if user == 'FaA' and phase == 'M':
        return (1-np.exp(-x)) * 10
    if user == 'FaA' and phase == 'E':
        return (1-np.exp(-x)) * 100
    if user == 'FaA' and phase == 'W':
        return (1-np.exp(-x)) * 150
    if user == 'NFaY' and phase == 'M':
        return (1-np.exp(-x)) * 70
    if user == 'NFaY' and phase == 'E':
        return (1-np.exp(-x)) * 50
    if user == 'NFaY' and phase == 'W':
        return (1-np.exp(-x)) * 100


def aggregate_function(x, user):
    return (5*(disaggregate_function(x, user, 'M')+disaggregate_function(x, user, 'E'))+4*disaggregate_function(x, user, 'W'))/14


def sample_aggregate(x, user):
    return (5*(sample_disaggregate(x, user, 'M')+sample_disaggregate(x, user, 'E'))+4*sample_disaggregate(x, user, 'W'))/14
    #return aggregate_function(x, user) + np.random.normal(0, 2)


def sample_disaggregate(x, user, phase):
    return disaggregate_function(x, user, phase) + np.random.normal(0, 2)
