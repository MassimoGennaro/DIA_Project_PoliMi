from environment.BudgetEnvironment import *
import numpy as np
from learners.Subcampaign_Learner import * 
from environment.Subcampaign import *
from knapsack.knapsack import Knapsack

n_arms = 5
min_budget = 1
max_budget = 5

labels = ['FaY','FaA','NFaY']
subcampaigns = [Subcampaign(l) for l in labels]
num_subcampaigns = len(subcampaigns)

T = 60
gpts_rewards_per_experiment = []

budgets = np.linspace(min_budget, max_budget, n_arms)

n_experiments = 1
gpts_rewards_per_experiment = []



for e in range(n_experiments):

    env = BudgetEnvironment(subcampaigns)
    s_learners = [Subcampaign_Learner(budgets,l) for l in labels]
    
    for t in range(T):
        
        estimations = []
        for s in s_learners:
            estimate = [s.pull_arms()]
            ##print(estimate)
            estimations.append(estimate)
            
        super_arm = Knapsack(max_budget, estimations).optimize()
        
        total_reward = 0
        
        # TO DO: Trovare un modo corretto per indicizzare le subcampaign 
        # (modificare l'attuale label)
        
        for (subcampaign_id, budget) in super_arm:
            
            reward = env.get_subcampaign(subcampaign_id).aggr_sample(budget)
            total_reward += sum(reward)

            s_learners[subcampaign_id].update(budget, reward)
