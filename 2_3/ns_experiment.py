from environment.CampaignEnvironment import *
from learners.NS_Subcampaign_Learner import NS_Subcampaign_Learner
from learners.Subcampaign_Learner import Subcampaign_Learner
from knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt


class NonStationaryExperiment:
    def __init__(self, max_budget=5.0, sigma=2.0, n_exp=10, horizon=56,
                 phase_weights=[5 / 14, 5 / 14, 4 / 14], sample_factor=4,
                 time_factor=1):
        # Budget settings
        self.max_budget = max_budget
        # Experiment settings
        self.sigma = sigma,
        self.n_experiments = n_exp
        self.T = horizon  # Time Horizon
        self.n_arms = int(self.max_budget + 1)
        self.budgets = np.linspace(0.0, self.max_budget, self.n_arms)
        # Phase settings
        self.phase_labels = ["Morning", "Evening", "Weekend"]
        self.phase_weights = phase_weights  # the sum must be equal to 1
        self.phase_list = (([self.phase_labels.index("Morning")]*sample_factor + [self.phase_labels.index("Evening")]*sample_factor)*5 +
                           [self.phase_labels.index("Weekend")]*4*sample_factor)*time_factor
        # Class settings
        self.feature_labels = ["Young-Familiar",
                               "Adult-Familiar", "Young-NotFamiliar"]
        # Contains the rewards for each experiment (each element is a list of T rewards)
        self.gpts_rewards_per_experiment = []
        self.SWgpts_rewards_per_experiment = []
        self.opt_rewards_per_experiment = []
        self.ran = False

    def run(self):
    ########################
    # Clairvoyant Solution #
    ########################

        opt_env = Campaign(self.budgets, phases=self.phase_labels,
                        weights=self.phase_weights, sigma=0.0)
        for feature_label in self.feature_labels:
            opt_env.add_subcampaign(label=feature_label)

        optimal_super_arm_reward_phase = []
        for phase in range(len(self.phase_labels)):
            real_values = opt_env.round_all(phase=phase)
            optimal_super_arm = knapsack_optimizer(real_values)

            optimal_super_arm_reward = 0
            for (subc_id, pulled_arm) in enumerate(optimal_super_arm):
                optimal_reward = opt_env.subcampaigns[subc_id].round(
                    pulled_arm, phase=phase)
                optimal_super_arm_reward += optimal_reward

            optimal_super_arm_reward_phase.append(optimal_super_arm_reward)

        for t in range(0, self.T):
            self.opt_rewards_per_experiment.append(
                optimal_super_arm_reward_phase[self.phase_list[t]])

        #########################
        # Experiment Solution #
        #########################

        for e in range(0, self.n_experiments):
            print("Performing experiment: ", str(e+1))

            # Create the BudgetEnvironment usint the list of sucampaigns
            env = Campaign(self.budgets, phases=self.phase_labels,
                        weights=self.phase_weights, sigma=self.sigma)

            # list of GP-learners
            subc_learners = []
            SW_s_learners = []

            # add subcampaings to the environment
            # and create a GP-learner for each subcampaign
            for feature_label in self.feature_labels:
                env.add_subcampaign(label=feature_label)
                # Non Sliding Windows
                subc_learners.append(Subcampaign_Learner(
                    arms=self.budgets, label=feature_label))
                # Sliding Window
                SW_s_learners.append(NS_Subcampaign_Learner(
                    arms=self.budgets, label=feature_label, horizon=self.T))
            
            sw_rewards = []
            rewards = [] 
            for t in range(0, self.T):

            ### SLIDING WINDOW ###

            # sample clicks estimations from GP-learners
            # and build the Knapsack table
                estimations = []
                for SW_s_learner in SW_s_learners:
                    estimate = SW_s_learner.pull_arms()

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
                    arm_reward = env.subcampaigns[subc_id].round(
                        pulled_arm, phase=self.phase_list[t])
                    super_arm_reward += arm_reward
                    SW_s_learners[subc_id].update(pulled_arm, arm_reward, t)

                # store the reward for this timestamp
                sw_rewards.append(super_arm_reward)
                
                ### NON SLIDING WINDOW ###

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
                    arm_reward = env.subcampaigns[subc_id].round(
                        pulled_arm, phase=self.phase_list[t])
                    super_arm_reward += arm_reward
                    subc_learners[subc_id].update(pulled_arm, arm_reward)

                # store the reward for this timestamp
                rewards.append(super_arm_reward)

            self.SWgpts_rewards_per_experiment.append(sw_rewards)
            self.gpts_rewards_per_experiment.append(rewards)

        self.ran = True

    def plot_experiment(self):
        if not self.ran:
            return "Run the experiment before plotting"

        plt.figure()
        plt.ylabel("Number of Clicks")
        plt.xlabel("t")

        mean_exp_SW = np.mean(self.SWgpts_rewards_per_experiment, axis=0)
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        opt = self.opt_rewards_per_experiment

        plt.plot(mean_exp_SW, 'b', label='Expected Reward SW')
        plt.plot(mean_exp, 'b--', label='Expected Reward no SW')
        plt.plot(opt, 'g', label='Optimal Reward')
        
        plt.legend(loc="upper left")
        plt.show()


    def plot_regret(self):
        plt.figure()
        plt.ylabel("Regret")
        plt.xlabel("t")
        
        mean_exp_SW = np.mean(self.SWgpts_rewards_per_experiment, axis=0)
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        opt = self.opt_rewards_per_experiment
        regret_SW = np.cumsum(opt - mean_exp_SW)
        regret = np.cumsum(opt - mean_exp)
        
        plt.plot(regret_SW, 'r', label='Regret SW')
        plt.plot(regret, 'r--', label='Regret no SW')
        
        plt.legend(loc="upper left")
        plt.show()
        

