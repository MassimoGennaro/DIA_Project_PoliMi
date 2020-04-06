import numpy as np


def disaggregate_function(x, user, phase):
    if user == 'FaY' and phase == 'M':
        return (1-np.exp(-x)) * 10
    if user == 'FaY' and phase == 'E':
        return (1-np.exp(-x)) * 15
    if user == 'FaY' and phase == 'W':
        return (1-np.exp(-x)) * 30
    if user == 'FaA' and phase == 'M':
        return (1-np.exp(-x)) * 1
    if user == 'FaA' and phase == 'E':
        return (1-np.exp(-x)) * 10
    if user == 'FaA' and phase == 'W':
        return (1-np.exp(-x)) * 15
    if user == 'NFaY' and phase == 'M':
        return (1-np.exp(-x)) * 7
    if user == 'NFaY' and phase == 'E':
        return (1-np.exp(-x)) * 5
    if user == 'NFaY' and phase == 'W':
        return (1-np.exp(-x)) * 10


def aggregate_function(x, user):
    return (5*(disaggregate_function(x, user, 'M')+disaggregate_function(x, user, 'E'))+2*disaggregate_function(x, user, 'W'))/7


def sample_aggregate(x, user):
    return aggregate_function(x, user) + np.random.normal(0, 2)


def sample_disaggregate(x, user, phase):
    return disaggregate_function(x, user, phase) + np.random.normal(0, 2)
