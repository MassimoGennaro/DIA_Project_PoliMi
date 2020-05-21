from .GPTS_Learner import *
import math


class NS_Subcampaign_Learner(GPTS_Learner):

    def __init__(self, arms, label, window_size):
        super().__init__(arms)
        self.label = label
        self.window_size = window_size #int(round(math.sqrt(horizon)))

    def pull_arms(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        sampled_values = np.maximum(0, sampled_values)  # avoid negative values
        return sampled_values

    def update_model(self, t):
        """
        if t <= window size is the same function of the stationay learner
        otherwise only the last 'window size' collected rewards and
        pulled arms are used by the fit function
        """
        if t <= self.window_size:
            x = np.atleast_2d(self.pulled_arms).T
            y = self.collected_rewards
            self.gp.fit(x, y)
            self.means, self.sigmas = self.gp.predict(
                np.atleast_2d(self.arms).T, return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-2)
        else:
            x = np.atleast_2d(self.pulled_arms[t-self.window_size:]).T
            y = self.collected_rewards[t-self.window_size:]
            self.gp.fit(x, y)
            self.means, self.sigmas = self.gp.predict(
                np.atleast_2d(self.arms).T, return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward, t):
        self.update_observations(pulled_arm, reward)
        self.update_model(t)
