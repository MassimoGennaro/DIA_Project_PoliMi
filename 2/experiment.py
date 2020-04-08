from environment.BudgetEnvironment import *
import numpy as np
from learners.Subcampaign_Learner import *
from environment.Subcampaign import *
from knapsack.knapsack import *

max_budget = 5
n_arms = max_budget+1

budgets = np.linspace(0, max_budget, n_arms)
labels = ['FaY', 'FaA', 'NFaY']

# Create a list of Subcampaign
subcampaigns = [Subcampaign(l) for l in labels]
num_subcampaigns = len(subcampaigns)

T = 60

n_experiments = 1
gpts_rewards_per_experiment = []

for e in range(n_experiments):
    # Create the BudgetEnvironment usint the list of sucampaigns
    env = BudgetEnvironment(subcampaigns)
    # Create a list of Subcampaign_GP
    s_learners = [Subcampaign_Learner(budgets, l) for l in labels]
    

    ## TESTING ##
    reward = env.get_subcampaign_by_idx(1).aggr_sample(1)
    #print(reward)
    print("Values of means and sigmas before the update: ", s_learners[0].means, s_learners[0].sigmas)
    s_learners[0].update(1, reward)
    print("Values of means and sigmas after the update:", s_learners[0].means, s_learners[0].sigmas)
    #############


    for t in range(T):
        # Sample from the Subcampaign_GP to get clicks estimations for each arm
        # and build the table to pass to Knapsack
        estimations = []
        for s in s_learners:
            estimate = [s.sample_from_GP(a) for a in budgets]
            # print(estimate)
            if(sum(estimate) == 0):
                estimate = [i * 1e-3 for i in range(n_arms)]
            estimations.append(estimate)
        # Knapsack return the super_arm as [(subcampaign, best_budget to assign), ..]
        super_arm = knapsack_optimizer(estimations) # Knapsack(max_budget, estimations).optimize()
        # RISOLTO?
        # Problema, knapsack ritorna valori di budget che non corrispondono a quelli interi ->
        # -> Viene poi generato un errore durante l'update
        print(super_arm)

        # For each subcampaign and budget related to it, use the budget to sample from the environment (click function)
        # Then use the sample obtained to update the current Subcampaing_GP
        for (subcampaign_id, budget_arm) in super_arm:
            reward = env.get_subcampaign_by_idx(subcampaign_id).aggr_sample(budget_arm)
            #print(reward)
            #total_reward = 0
            #total_reward += sum(reward)
            s_learners[subcampaign_id].update(budget_arm, reward)
