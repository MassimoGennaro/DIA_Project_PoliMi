import numpy as np
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTS_Learner:
    def __init__(self, arms):
        
        # max_arm = np.max(arms)
        # min_arm = np.min(arms)
        # normalize_arms = (arms-min_arm)/(max_arm - min_arm)
        self.arms = arms
        
        self.n_arms = len(arms)

        # self.t = 0
        # self.rewards_per_arm = [[] for i in range(self.n_arms)]
        self.collected_rewards = np.array([])
        self.pulled_arms = []

        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10

        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2,
                                                                    normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, pulled_arm, reward):
        """
        Updates what the Gaussian Process has observed so far in two lists,
        collected_rewards and pulled arms
        """
        # self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        """
        Updates the Gaussian Process using the pulled arms and their rewards.
        """
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(
            np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        """
        Call the two functions above
        """
        # self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, arm_idx):
        """
        Returns the predictions of the arms computing a 
        normal distribution for each of them using the currently means and sigmas.
        Note that it return one or more values as sample_values, depending on the
        length of means and sigmas
        """
        sampled_value = np.random.normal(
            self.means[arm_idx], self.sigmas[arm_idx])
        # to avoid negative values
        sampled_value = np.maximum(0, sampled_value)
        return sampled_value

    def find_arm(self, arm):
        """
        Return the index of the given arm contained in self.arms if exixsts,
        False otherwise
        """
        for idx in range(self.n_arms):
            if self.arms[idx] == arm:
                return idx
        return False

    def learn_kernel_hyperparameters(self, samples):
        x = np.atleast_2d(samples[0]).T
        y = [y for (x, y) in zip(samples[0], samples[1])]
        self.gp.fit(x, y)
        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=self.gp.kernel_,
                                                                    alpha=self.gp.alpha,
                                                                    normalize_y=True, n_restarts_optimizer=0)