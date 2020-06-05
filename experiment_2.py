from Advertising.environment.CampaignEnvironment import *
from Advertising.environment.Advertising_Config_Manager import *
from Advertising.learners.Subcampaign_Learner import *
from Advertising.knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt

class Experiment_2:
    def __init__(self, max_budget=5.0, n_arms=6, env_id=0):
        # Budget settings
        self.max_budget = max_budget
        self.n_arms = n_arms
        self.budgets = np.linspace(0.0, self.max_budget, self.n_arms)

        env = Advertising_Config_Manager(env_id)
        self.real_values = None

        # Phase settings
        self.phase_labels = env.phase_labels
        self.phase_weights = env.get_phase_weights()

        # Class settings
        self.feature_labels = env.feature_labels

        # Click functions
        self.click_functions = env.click_functions

        # Experiment settings
        self.sigma = env.sigma

        self.opt_super_arm_reward = None # self.run_clairvoyant()

        ## Rewards for each experiment (each element is a list of T rewards)
        self.opt_rewards_per_experiment = []
        self.gpts_rewards_per_experiment = []

        self.ran = False


    def run_clairvoyant(self):
        """
        Clairvoyant Solution
        :return: list of optimal super-arm reward for each phase
        """

        opt_env = Campaign(self.budgets, phases=self.phase_labels, weights=self.phase_weights)
        for feature_label in self.feature_labels:
            opt_env.add_subcampaign(label=feature_label, functions=self.click_functions[feature_label])

        real_values = opt_env.round_all()
        self.real_values = real_values
        opt_super_arm = knapsack_optimizer(real_values)

        opt_super_arm_reward = 0
        for (subc_id, pulled_arm) in enumerate(opt_super_arm):
            reward = opt_env.subcampaigns[subc_id].round(pulled_arm)
            opt_super_arm_reward += reward

        self.opt_super_arm_reward = opt_super_arm_reward

        return get_dataframe(real_values, opt_super_arm, self.budgets)


    def run(self, n_experiments=10, horizon=56, GP_graphs=False):
        """
        Experimental Solution
        :return:
        """

        assert (self.opt_super_arm_reward is not None), "Run the clairvoyant solution before!"

        self.gpts_rewards_per_experiment = []
        self.opt_rewards_per_experiment = [self.opt_super_arm_reward] * horizon

        for e in range(0, n_experiments):
            print("Performing experiment: ", str(e + 1))

            # create the environment
            env = Campaign(self.budgets, phases=self.phase_labels, weights=self.phase_weights, sigma=self.sigma)

            # list of GP-learners
            subc_learners = []
            
            # add subcampaings to the environment
            # and create a GP-learner for each subcampaign
            # max_budget = self.max_budget
            # min_budget = np.min(self.budgets)
            # norm_budgets = (self.budgets-min_budget)/(max_budget - min_budget)
            for subc_id, feature_label in enumerate(self.feature_labels):
                env.add_subcampaign(label=feature_label, functions=self.click_functions[feature_label])
                learner = Subcampaign_Learner(arms=self.budgets, label=feature_label)
                
                clicks = env.subcampaigns[subc_id].round_all()
                samples = [self.budgets, clicks]
                learner.learn_kernel_hyperparameters(samples)
                subc_learners.append(learner)

            # rewards for each time step
            rewards = []

            for t in range(0, horizon):
                # sample clicks estimations from GP-learners
                # and build the Knapsack table
                estimations = []
                for subc_learner in subc_learners:
                    estimate = subc_learner.pull_arms()

                    # force 0 clicks for budget equal to 0
                    estimate[0] = 0

                    """if(sum(estimate) == 0):
                        estimate = [i * 1e-3 for i in range(n_arms)]"""

                    estimations.append(estimate)

                # Knapsack return a list of pulled_arm
                super_arm = knapsack_optimizer(estimations)

                super_arm_reward = 0

                # sample the number of clicks from the environment
                # and update the GP-learners in the pulled arms
                for (subc_id, pulled_arm) in enumerate(super_arm):
                    arm_reward = env.subcampaigns[subc_id].round(pulled_arm)
                    super_arm_reward += arm_reward
                    subc_learners[subc_id].update(pulled_arm, arm_reward)

                # store the reward for this timestamp
                rewards.append(super_arm_reward)

            self.gpts_rewards_per_experiment.append(rewards)

            if GP_graphs:
                self.plot_GP_graphs(subc_learners)

        self.ran = True

    def plot_experiment(self):
        if not self.ran:
            return "Run the experiment before plotting!"

        plt.figure()
        plt.ylabel("Number of Clicks")
        plt.xlabel("t")

        opt_exp = self.opt_rewards_per_experiment
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)

        plt.plot(opt_exp, 'g', label='Optimal Reward')
        plt.plot(mean_exp, 'b', label='Expected Reward')

        plt.legend(loc="upper left")
        plt.show()

    def plot_regret(self):
        if not self.ran:
            return 'Run the experiment before plotting'

        plt.figure()
        plt.ylabel("Regret")
        plt.xlabel("t")

        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        regret = np.cumsum(self.opt_super_arm_reward - mean_exp)

        plt.plot(regret, 'r', label='Regret')
        plt.legend(loc="upper left")
        plt.show()

    def plot_GP_graphs(self, subc_learners):

        x_pred = np.atleast_2d(self.budgets).T
        for i, subc_learner in enumerate(subc_learners):
            y_pred = subc_learner.means
            sigma = subc_learner.sigmas
            X = np.atleast_2d(subc_learner.pulled_arms).T
            Y = subc_learner.collected_rewards.ravel()
            real_values = self.real_values[i]
            title = subc_learner.label

            plt.plot(x_pred, real_values, 'r:', label=r'$click function$')
            plt.plot(X.ravel(), Y, 'ro', label=u'Observed Clicks')
            plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% conf interval')
            plt.title(title)
            plt.xlabel('$budget$')
            plt.ylabel('$daily clicks$')
            plt.legend(loc='lower right')
            plt.show()
