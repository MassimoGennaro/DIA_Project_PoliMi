from environment.ClickFunction import *


class Subcampaign():
    def __init__(self, label):
        self.label = label

    def aggr_sample(self, budget):
        sample_aggregate(budget, self.label)

    def disaggr_sample(self, budget, phase):
        sample_disaggregate(budget, self.label, phase)