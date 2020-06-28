from Advertising.environment.Advertising_Config_Manager import *
from Advertising.environment.CampaignEnvironment import *
from Advertising.learners.NS_Subcampaign_Learner import NS_Subcampaign_Learner
from Advertising.learners.Subcampaign_Learner import Subcampaign_Learner
from Advertising.knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt

class Experiment_3:
    def __init__(self, max_budget=5.0, n_arms=6, sample_factor=8, env_id=0, estimate_hyperparam = False):
        """
        <description>
        :param max_budget: maximal value of budget
        :param n_arms: number of arms
        :param sample_factor: number of samples for each sub-phase (here, semi-day)
        """

        # Budget settings
        self.max_budget = max_budget
        self.n_arms = n_arms
        self.budgets = np.linspace(0.0, self.max_budget, self.n_arms)

        env = Advertising_Config_Manager(env_id)

        # Phase settings
        self.phase_labels = env.phase_labels
        self.phase_weights = env.get_phase_weights()
        self.phase_list = env.get_phase_list(sample_factor)
        self.phase_len = len(self.phase_list)

        # Class settings
        self.feature_labels = env.feature_labels

        # Click functions
        self.click_functions = env.click_functions

        # Experiment settings
        self.sigma = env.sigma
        self.estimate_hyperparam = estimate_hyperparam

        self.optimal_super_arm_reward_phase = self.run_clairvoyant()

        # Rewards for each experiment (each element is a list of T rewards)
        self.opt_rewards_per_experiment = []
        self.gpts_rewards_per_experiment = []
        self.SWgpts_rewards_per_experiment = []

        self.ran = False
        self.window_size = None


    def run_clairvoyant(self):
        """
        Clairvoyant Solution
        :return: list of optimal super-arm reward for each phase
        """

        opt_env = Campaign(self.budgets, phases=self.phase_labels, weights=self.phase_weights)
        for feature_label in self.feature_labels:
            opt_env.add_subcampaign(label=feature_label, functions=self.click_functions[feature_label])

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

        return optimal_super_arm_reward_phase

    def run(self, n_experiments=10, horizon=56, window_size=6):
        """
        Experimental Solution
        :return:
        """

        # optimal reward per experiment
        self.opt_rewards_per_experiment = []
        for t in range(0, horizon):
            self.opt_rewards_per_experiment.append(
                self.optimal_super_arm_reward_phase[self.phase_list[t % self.phase_len]])

        self.SWgpts_rewards_per_experiment = []
        self.gpts_rewards_per_experiment = []
        for e in range(0, n_experiments):
            print("Performing experiment: ", str(e + 1))

            # Create the environment
            env = Campaign(self.budgets, phases=self.phase_labels, weights=self.phase_weights, sigma=self.sigma)

            # list of GP-learners
            subc_learners = []
            SW_s_learners = []

            # add subcampaigns to the environment
            # and create a GP-learner for each subcampaign
            for subc_id, feature_label in enumerate(self.feature_labels):
                env.add_subcampaign(label=feature_label, functions=self.click_functions[feature_label])
                
                if self.estimate_hyperparam == True:
                    clicks = env.subcampaigns[subc_id].round_all(phase=0) #Phase = None -> aggregates the phases of the click_function
                    samples = [self.budgets, clicks]
                    
                    learner = Subcampaign_Learner(arms=self.budgets, label=feature_label)
                    learner.learn_kernel_hyperparameters(samples)
                    subc_learners.append(learner)
                    
                    sw_learner = NS_Subcampaign_Learner(arms=self.budgets, label=feature_label, window_size=window_size)
                    sw_learner.learn_kernel_hyperparameters(samples)
                    SW_s_learners.append(sw_learner)
                else:
                    # Non Sliding Windows
                    subc_learners.append(Subcampaign_Learner(arms=self.budgets, label=feature_label))
                    # Sliding Window
                    SW_s_learners.append(NS_Subcampaign_Learner(arms=self.budgets, label=feature_label, window_size=window_size))

            sw_rewards = []
            rewards = []
            for t in range(0, horizon):

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
                        pulled_arm, phase=self.phase_list[t % self.phase_len])
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
                        pulled_arm, phase=self.phase_list[t % self.phase_len])
                    super_arm_reward += arm_reward
                    subc_learners[subc_id].update(pulled_arm, arm_reward)

                # store the reward for this timestamp
                rewards.append(super_arm_reward)

            self.SWgpts_rewards_per_experiment.append(sw_rewards)
            self.gpts_rewards_per_experiment.append(rewards)

        self.ran = True

    def multiple_run(self, n_experiments=10, horizon=56, window_size=[4,6,8]):
        """
        Experimental Solution
        :return:
        """

        # optimal reward per experiment
        self.opt_rewards_per_experiment = []
        for t in range(0, horizon):
            self.opt_rewards_per_experiment.append(
                self.optimal_super_arm_reward_phase[self.phase_list[t % self.phase_len]])

        self.SWgpts_rewards_per_experiment = []
        self.gpts_rewards_per_experiment = []
        for e in range(0, n_experiments):
            print("Performing experiment: ", str(e + 1))

            # Create the environment
            env = Campaign(self.budgets, phases=self.phase_labels, weights=self.phase_weights, sigma=self.sigma)

            # list of GP-learners
            subc_learners = []
            SW_s_learners = [[] for i in range(len(window_size))]

            # add subcampaigns to the environment
            # and create a GP-learner for each subcampaign
            for subc_id, feature_label in enumerate(self.feature_labels):
                env.add_subcampaign(label=feature_label, functions=self.click_functions[feature_label])

                if self.estimate_hyperparam == True:
                    clicks = env.subcampaigns[subc_id].round_all(
                        phase=None)  # Phase = None -> aggregates the phases of the click_function
                    samples = [self.budgets, clicks]

                    learner = Subcampaign_Learner(arms=self.budgets, label=feature_label)
                    learner.learn_kernel_hyperparameters(samples)
                    subc_learners.append(learner)

                    for w in range(len(window_size)):
                        sw_learner = NS_Subcampaign_Learner(arms=self.budgets, label=feature_label, window_size=window_size[w])
                        sw_learner.learn_kernel_hyperparameters(samples)
                        SW_s_learners[w].append(sw_learner)
                else:
                    # Non Sliding Windows
                    subc_learners.append(Subcampaign_Learner(arms=self.budgets, label=feature_label))
                    # Sliding Window
                    for w in range(len(window_size)):
                        SW_s_learners[w].append(
                            NS_Subcampaign_Learner(arms=self.budgets, label=feature_label, window_size=window_size[w]))

            sw_rewards = [[] for i in range(len(window_size))]
            rewards = []
            for t in range(0, horizon):
                ### SLIDING WINDOW ###
                for w in range(len(window_size)):
                    # sample clicks estimations from GP-learners
                    # and build the Knapsack table
                    estimations = []
                    for SW_s_learner in SW_s_learners[w]:
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
                            pulled_arm, phase=self.phase_list[t % self.phase_len])
                        super_arm_reward += arm_reward
                        SW_s_learners[w][subc_id].update(pulled_arm, arm_reward, t)

                    # store the reward for this timestamp
                    sw_rewards[w].append(super_arm_reward)

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
                        pulled_arm, phase=self.phase_list[t % self.phase_len])
                    super_arm_reward += arm_reward
                    subc_learners[subc_id].update(pulled_arm, arm_reward)

                # store the reward for this timestamp
                rewards.append(super_arm_reward)

            self.SWgpts_rewards_per_experiment.append(sw_rewards)
            self.gpts_rewards_per_experiment.append(rewards)

        self.ran = True
        self.window_size = window_size

    def plot_experiment(self):
        if not self.ran:
            return "Run the experiment before plotting"

        plt.figure()
        plt.ylabel("Number of Clicks")
        plt.xlabel("t")

        opt_exp = self.opt_rewards_per_experiment
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        mean_exp_SW = np.mean(self.SWgpts_rewards_per_experiment, axis=0)

        plt.plot(opt_exp, 'g', label='Optimal Reward')
        plt.plot(mean_exp, 'b--', label='Expected Reward no SW')
        plt.plot(mean_exp_SW, 'b', label='Expected Reward SW')

        plt.legend(loc="upper left")
        plt.show()

    def plot_regret(self):
        if not self.ran:
            return "Run the experiment before plotting"

        plt.figure()
        plt.ylabel("Regret")
        plt.xlabel("t")

        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        mean_exp_SW = np.mean(self.SWgpts_rewards_per_experiment, axis=0)
        opt_exp = self.opt_rewards_per_experiment

        regret_SW = np.cumsum(opt_exp - mean_exp_SW)
        regret = np.cumsum(opt_exp - mean_exp)

        plt.plot(regret_SW, 'r', label='Regret SW')
        plt.plot(regret, 'r--', label='Regret no SW')

        plt.legend(loc="upper left")
        plt.show()

    def plot_multiple_regret(self):
        if not self.ran:
            return "Run the experiment before plotting"

        plt.figure()
        plt.ylabel("Regret")
        plt.xlabel("t")

        opt_exp = self.opt_rewards_per_experiment
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        regret = np.cumsum(opt_exp - mean_exp)
        regret_SW = []
        for w in range(len(self.SWgpts_rewards_per_experiment[0])):
            temp = [i[w] for i in self.SWgpts_rewards_per_experiment]
            mean_exp_SW = np.mean(temp, axis=0)
            regret_SW.append(np.cumsum(opt_exp - mean_exp_SW))

        num_plots = len(regret_SW)

        labels = []
        plt.plot(regret, 'r--')
        labels.append("no SW")
        for w in range(num_plots):
            plt.plot(regret_SW[w])
            labels.append("SW("+str(self.window_size[w])+")")

        """plt.legend(labels, ncol=4, loc='upper left',
                   bbox_to_anchor=[0.5, 1.1],
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)"""

        plt.legend(labels, loc="upper left")
        plt.show()