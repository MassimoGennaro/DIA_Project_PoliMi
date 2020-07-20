import numpy as np


class Personalized_Environment():
    """
    it contains info about sui cadnidates and theirs probabilities.
    """

    def __init__(self, arms_candidates, probabilities):
        # arms_candidates è array dei valori di ogni arm (e.g. [5, 10 ,15, 20 ,25])
        self.arms_candidates = arms_candidates
        # probabilities è un tensore in 3 dimensioni: [phase][category][arm]
        self.probabilities = probabilities
        self.time = 0


   
    def round(self, p_category, pulled_arm):
         # returns the reward of the arm chosen 

        p = self.probabilities[p_category][pulled_arm]
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward * self.arms_candidates[pulled_arm]
