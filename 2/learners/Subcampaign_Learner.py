from learners.GPTS_Learner import GPTS_Learner


class Subcampaign_Learner(GPTS_Learner):
    def __init__(self, arms, label):
        super().__init__(arms)
        self.label = label

