import json
import numpy as np


class Environment():
    def __init__(self, id):
        self.id = id
        with open('utils/sub_camp_config.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]
        self.sigma = campaign["sigma"]
        self.phase_labels = campaign["phases"]
        self.phase_weights = list(campaign["phase_weights"].values())
        self.subcampaigns = campaign["subcampaigns"]
        self.feature_labels = list(self.subcampaigns.keys())
        self.click_functions = self.create_functions()
            
    
    def create_functions(self):

        click_functions = {}

        for f in self.feature_labels:
            click_functions[f] = {}
            for i,p in enumerate(self.phase_labels):
                speed = self.subcampaigns[f][i]["speed"]
                max_value = self.subcampaigns[f][i]["max_value"]
                l = lambda x: (1 - np.exp(-(speed)*x)) * max_value
                click_functions[f][p] = l
        

        return click_functions
