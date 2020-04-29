import numpy as np


class Campaign:
    def __init__(self, budgets, phases, weights, sigma, click_functions):
        self.budgets = budgets

        self.phases = phases
        self.weights = weights

        self.subcampaigns = []
        self.sigma = sigma
        self.click_functions = click_functions

    def get_functions(self, label):
        print([self.click_functions[label][phase] for phase in self.phases])
        return [self.click_functions[label][phase] for phase in self.phases]

    def add_subcampaign(self, label):
        self.subcampaigns.append(
            Subcampaign(label, self.budgets, self.get_functions(label), self.sigma, self.weights)
        )

    # round a specific arm
    def round(self, subcampaign_id, pulled_arm, phase=None):
        return self.subcampaigns[subcampaign_id].round(pulled_arm, phase)

    # round all arms
    def round_all(self, phase=None):
        table = []
        for subcampaign in self.subcampaigns:
            table.append(subcampaign.round_all(phase))
        return table


class Subcampaign:
    def __init__(self, label, budgets, functions, sigma, weights):
        self.label = label
        self.weights = weights
        self.budgets = budgets
        self.n_phases = len(functions)
        self.phases = [Subphase(budgets, functions[i], sigma) for i in range(self.n_phases)]

    # round a specific arm
    def round(self, pulled_arm, phase=None):
        # aggregate sample
        if phase is None:
            return sum(self.weights[i] * self.phases[i].round(pulled_arm) for i in range(self.n_phases))
        # disaggregate sample
        else:
            return self.phases[phase].round(pulled_arm)

    # round all arms
    def round_all(self, phase=None):
        return [self.round(pulled_arm, phase) for pulled_arm in range(len(self.budgets))]


class Subphase:
    def __init__(self, budgets, function, sigma):
        self.means = function(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
