from learners.GPTS_Learner import GPTS_Learner


class Subcampaign_Learner(GPTS_Learner):
    def __init__(self, arms, label):
        super().__init__(arms)
        self.label = label
        

    def sample_from_GP(self, arm):
        """
        Sample from the GP with the given value of budget
        """
        arm_idx = self.find_arm(arm)
        return self.pull_arm(arm_idx)