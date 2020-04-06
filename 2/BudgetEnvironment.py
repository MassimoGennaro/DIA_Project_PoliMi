from ClickFunction import *


class BudgetEnvironment():
    def __init__(self, budgets):
        self.budgets = budgets

    def round(self, pulled_arm, user, phase=None):
        if phase is None:
            return sample_aggregate(pulled_arm, user)
        else:
            return sample_disaggregate(pulled_arm, user, phase)
