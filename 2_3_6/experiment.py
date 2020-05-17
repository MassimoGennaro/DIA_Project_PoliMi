from environment.CampaignEnvironment import *
from learners.Subcampaign_Learner import *
from knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, max_budget=5.0, n_arms=6, env_id=0):
        # Budget settings
        self.max_budget = max_budget
        self.n_arms = n_arms
        self.budgets = np.linspace(0.0, self.max_budget, self.n_arms)

        env = Environment(env_id)

        # Phase settings
        self.phase_labels = env.phase_labels
        self.phase_weights = env.get_phase_weights()

        # Class settings
        self.feature_labels = env.feature_labels

        # Click functions
        self.click_functions = env.click_functions

        # Experiment settings
        self.sigma = env.sigma

        self.opt_super_arm_reward = self.run_clairvoyant()

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
        opt_super_arm = knapsack_optimizer(real_values)

        opt_super_arm_reward = 0
        for (subc_id, pulled_arm) in enumerate(opt_super_arm):
            reward = opt_env.subcampaigns[subc_id].round(pulled_arm)
            opt_super_arm_reward += reward

        return opt_super_arm_reward


    def run(self, n_experiments=10, horizon=56):
        """
        Experimental Solution
        :return:
        """
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

        self.ran = True

    def plot_experiment(self):
        if not self.ran:
            return "Run the experiment before plotting"

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
