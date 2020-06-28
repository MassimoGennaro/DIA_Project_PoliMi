import json
import numpy as np


class Advertising_Config_Manager:
    def __init__(self, id):
        self.id = id
        with open('Advertising/configs/sub_camp_config.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]

        # Phase settings
        self.phase_labels = list(campaign["phases"])
        self.phase_seq = list(campaign["phase_seq"])

        # Class settings
        self.feature_labels = list(campaign["subcampaigns"].keys())

        # Experiment settings
        self.sigma = campaign["sigma"]
        self.click_functions = {}
        for feature in campaign["subcampaigns"]:
            self.click_functions[feature] = []
            for phase in campaign["subcampaigns"][feature]:
                max_value = phase["max_value"]
                speed = phase["speed"]

                assert (max_value >= 0), "Max value not valid for "+feature+": "+str(max_value)
                assert (0 <= speed <= 1), "Speed value not valid for "+feature+": "+str(speed)

                self.click_functions[feature].append( lambda x, s=speed, m=max_value: self.function(x, s, m) )

            assert (len(self.click_functions[feature])==len(self.phase_labels)), "Not a valid number of subphases for " + feature

        """self.click_functions = {
            "Young-Familiar": [
                lambda x: (1 - np.exp(-x/s)) * m,
                lambda x: (1 - np.exp(-x)) * 150,
                lambda x: (1 - np.exp(-x)) * 300
            ],
            "Adult-Familiar": [
                lambda x: (1 - np.exp(-x)) * 10,
                lambda x: (1 - np.exp(-x)) * 100,
                lambda x: (1 - np.exp(-x)) * 150
            ],
            "Young-NotFamiliar": [
                lambda x: (1 - np.exp(-x)) * 70,
                lambda x: (1 - np.exp(-x)) * 50,
                lambda x: (1 - np.exp(-x)) * 100
            ],
        }"""

    def function(self, x, s, m):
        return (1 - np.exp(-s*x)) * m

    def get_phase_weights(self):
        a = np.array(self.phase_seq)
        _, counts = np.unique(a, return_counts=True)
        mcd = sum(counts)
        return [w / mcd for w in counts]  # normalization

    def get_phase_list(self, sample_factor):
        phase_list = []
        for i in range(len(self.phase_seq)):
            phase_list += [self.phase_seq[i]]*sample_factor
        return phase_list
