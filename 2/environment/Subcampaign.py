from environment.ClickFunction import *


class Subcampaign():
    def __init__(self, label):
        self.label = label

    def aggr_sample(self, budget):
        return sample_aggregate(budget, self.label)

    def disaggr_sample(self, budget, phase):
        return sample_disaggregate(budget, self.label, phase)