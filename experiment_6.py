from Advertising.environment.Advertising_Config_Manager import *
from Advertising.environment.CampaignEnvironment import *
from Advertising.learners.Subcampaign_Learner import *
from Advertising.knapsack.knapsack import *
from Pricing.environment.PricingEnvironment import *
from Pricing.environment.Pricing_Config_Manager import *
from Pricing.modules.Experiment_Manager import *
from Pricing.modules.Person_Manager import *
from Pricing.modules.Context_Manager import *

import numpy as np
import matplotlib.pyplot as plt

class Experiment_6:
    def __init__(self, max_budget=5.0, n_arms=6, pricing_env_id = 0, advertising_env_id = 0):
        ## ADVERTISING ##
        
        # Budget settings
        self.max_budget = max_budget
        self.n_arms = n_arms
        self.budgets = np.linspace(0.0, self.max_budget, self.n_arms)

        adv_env = Advertising_Config_Manager(advertising_env_id)

        # Phase settings
        self.phase_labels = adv_env.phase_labels
        self.phase_weights = adv_env.get_phase_weights()

        # Class settings
        self.feature_labels = adv_env.feature_labels

        # Click functions
        self.click_functions = adv_env.click_functions
        
        # Experiment settings
        self.sigma = adv_env.sigma
        ################
        
        ## PRICING ##

        pri_env = Pricing_Config_Manager(pricing_env_id)
        self.categories = pri_env.get_indexed_categories()
        self.features = pri_env.features
        self.features_space = pri_env.feature_space
        self.p_categories = np.array(pri_env.probabilities)
        self.arms_candidates = np.array(pri_env.prices)
        self.n_arms_price = len(self.arms_candidates)
        ################
        

        
        ## Clairvoyant optimal reward ##
        self.opt_super_arm_reward = self.run_clairvoyant()

        ## Rewards for each experiment (each element will be a list of 'horizon' rewards)
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

        real_click_values = opt_env.round_all()
          
        expected_values = [[a*b for a,b in zip(self.arms_candidates, self.p_categories[i])] for i in range(len(self.p_categories))]
        
        real_expected_values = [max(expected_values[i]) for i in range(len(expected_values))]
        
        real_values = [[real_expected_values[a]*real_click_values[a][b] for b in range(self.n_arms)] for a in range(len(real_expected_values))]
        opt_super_arm = knapsack_optimizer(real_values)

        opt_super_arm_reward = 0
        for (subc_id, pulled_arm) in enumerate(opt_super_arm):
            reward = opt_env.subcampaigns[subc_id].round(pulled_arm) * real_expected_values[subc_id]
            opt_super_arm_reward += reward

        return opt_super_arm_reward


    def create_general(self):
        
        
        environment = Personalized_Environment(self.arms_candidates, self.p_categories)
        
        p_manager = Person_Manager(self.categories, self.p_categories, self.features)
        
        c_manager = Context_Manager(self.n_arms_price, self.features_space, self.categories, self.arms_candidates, contexts_known=True)
        
        # c_manager.add_context() crea un contesto della categoria passata
        for i in range(len(self.categories)):
            
            c_manager.add_context(self.categories[i],print_init=False)
            c_manager.add_context(self.categories[i],print_init=False)
            c_manager.add_context(self.categories[i],print_init=False)
        
        # general gestisce la logica
        general = General(p_manager, c_manager, environment)
        
        return general
    
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
            
            # create 'general' object that handles the pricing part
            pricing_experiment = self.create_general()
            
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
                click_estimations = []
                for subc_learner in subc_learners:
                    estimate = subc_learner.pull_arms()

                    # force 0 clicks for budget equal to 0
                    estimate[0] = 0

                    """if(sum(estimate) == 0):
                        estimate = [i * 1e-3 for i in range(n_arms)]"""

                    click_estimations.append(estimate)
                '''
                Here we need to multiply each cell of click_estimations for the (estimated) best expected value of each class, this values come from the pricing algorithm
                and then pass the updated table to knapsack
                '''
                expected_values = pricing_experiment.expected_values #[3 x 5]
                
                
                best_exp_values = [max(expected_values[i]) for i in range(len(expected_values))]
                
                values = [[best_exp_values[a]*click_estimations[a][b] for b in range(self.n_arms)] for a in range(len(expected_values))]
                
                
                # Knapsack return a list of pulled_arm
                super_arm = knapsack_optimizer(values)
               
                '''
                We need to extract the original number of click corresponding to the selected budget of a given class and give them to the pricing algorithm
                '''
                best_n_clicks = [] # this list goes to pricing
                
                
                super_arm_reward = 0
                # sample the number of clicks from the environment
                # and update the GP-learners in the pulled arms
                for (subc_id, pulled_arm) in enumerate(super_arm):
                    #The sampling from the environment and update of the "budget learners" remains the same
                    arm_reward = env.subcampaigns[subc_id].round(pulled_arm)
                    subc_learners[subc_id].update(pulled_arm, arm_reward)
                    best_n_clicks.append(arm_reward)
                    
                
                pricing_experiment.run_pricing_experiment(best_n_clicks)
                
                exp_values = pricing_experiment.expected_values
                
                best_exp_values = [max(exp_values[i]) for i in range(len(exp_values))]
                
                # store the reward for this timestamp
                super_arm_reward = [(c * e) for c, e in zip(best_n_clicks, best_exp_values)]
                rewards.append(sum(super_arm_reward))

            self.gpts_rewards_per_experiment.append(rewards)
            
        self.ran = True

    def plot_experiment(self):
        if not self.ran:
            return "Run the experiment before plotting"

        plt.figure()
        plt.ylabel("Reward (n_clicks * Value per click)")
        plt.xlabel("t")
        
        opt_exp = self.opt_rewards_per_experiment
        plt.plot(opt_exp, 'g', label='Optimal Reward')
        mean_exp = np.mean(self.gpts_rewards_per_experiment, axis=0)
        plt.plot(mean_exp, 'b', label='Expected Reward')
        

        plt.legend(loc="lower right")
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
        plt.legend(loc="lower right")
        plt.show()
