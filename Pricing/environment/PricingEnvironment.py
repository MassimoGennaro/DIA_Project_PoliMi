import numpy as np

##### ENVIRONMENT #####
# contiene informazioni sui candidati e probabilità.
# le probabilità di ogni categoria possono variare se variano le fasi

# quando t avanza di tempo, può cambiare fase.
class Personalized_Environment():
    """

    """

    def __init__(self, arms_candidates, probabilities):
        # arms_candidates è array dei valori di ogni arm (e.g. [5, 10 ,15, 20 ,25])
        self.arms_candidates = arms_candidates
        # probabilities è un tensore in 3 dimensioni: [phase][category][arm]
        self.probabilities = probabilities
        self.time = 0


    # rende la reward del candidato in base alla [phase][category][arm]
    def round(self, p_category, pulled_arm):

        p = self.probabilities[p_category][pulled_arm]
        reward = np.random.binomial(1, p)
        self.time += 1
        return reward * self.arms_candidates[pulled_arm]
