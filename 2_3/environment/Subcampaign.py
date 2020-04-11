from environment.ClickFunction import *


class Subcampaign():
    def __init__(self, label):
        self.label = label

    def aggr_sample(self, budget):
        return sample_aggregate(budget, self.label)

    def disaggr_sample(self, budget, phase):
        return sample_disaggregate(budget, self.label, phase)

    def real_function_aggr(self, budget):
        return aggregate_function(budget, self.label)

    def real_function_disaggr(self, budget, phase):
        return disaggregate_function(budget, self.label, phase)
