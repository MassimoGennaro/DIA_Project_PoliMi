from learners.GPTS_Learner import GPTS_Learner
import numpy as np

class Subcampaign_Learner(GPTS_Learner):
    def __init__(self, arms, label):
        super().__init__(arms)
        self.label = label

    def pull_arms(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        sampled_values = np.maximum(0, sampled_values)  # avoid negative values
        return sampled_values