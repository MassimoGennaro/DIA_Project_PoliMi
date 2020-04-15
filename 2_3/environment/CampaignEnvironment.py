class BudgetEnvironment():
    def __init__(self, subcampaigns):
        self.subcampaigns = subcampaigns
        # Dict to link subcampaings label with index
        self.link = {i:self.subcampaigns[i] for i in range(len(self.subcampaigns))} 

    def get_subcampaign_by_label(self, label):
        return self.subcampaigns.index(label)
        
    def get_subcampaign_by_idx(self, idx):
        return self.link[idx]