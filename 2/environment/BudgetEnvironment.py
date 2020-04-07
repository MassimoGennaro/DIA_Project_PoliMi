class BudgetEnvironment():
    def __init__(self, subcampaigns):
        self.subcampaigns = subcampaigns

    def get_subcampaign(self, label):
        return self.subcampaigns.index(label)
        
