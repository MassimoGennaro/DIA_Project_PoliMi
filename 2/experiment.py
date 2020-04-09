from environment.BudgetEnvironment import *
import numpy as np
from learners.Subcampaign_Learner import *
from environment.Subcampaign import *
from knapsack.knapsack import *
import matplotlib.pyplot as plt

max_budget = 5
n_arms = max_budget + 1

budgets = np.linspace(0, max_budget, n_arms)
labels = ['FaY', 'FaA', 'NFaY']

# Create a list of Subcampaign
subcampaigns = [Subcampaign(l) for l in labels]
num_subcampaigns = len(subcampaigns)

# Phases
T = 56
phase_1_ts = [t for t in range(T-16) if (t % 8) < 4]
phase_2_ts = [t for t in range(T-16) if (t % 8) >= 4]
phase_3_ts = [t for t in range(T-16, T)]


n_experiments = 10


# Optimal Solution

env_opt = BudgetEnvironment(subcampaigns)

optimal_super_arm_reward = 0
real_values = []
for subc in env_opt.subcampaigns:
    real_values.append([subc.real_function_aggr(b) for b in budgets])
    optimal_super_arm = knapsack_optimizer(real_values)
   #print("Optimal superarm:", optimal_super_arm)
for (subcampaign_id, budget_arm) in optimal_super_arm:
    optimal_reward = env_opt.get_subcampaign_by_idx(
    subcampaign_id).real_function_aggr(budget_arm)
    optimal_super_arm_reward += optimal_reward
 #print("Value of optimal superarm = ", optimal_super_arm_reward)
opt_rewards_per_experiment = optimal_super_arm_reward


# Contains the rewards for each experiment (each element is a list of T rewards)
gpts_rewards_per_experiment = []

for e in range(n_experiments):
    # Create the BudgetEnvironment usint the list of sucampaigns
    env = BudgetEnvironment(subcampaigns)

    # Create a list of Subcampaign_GP
    s_learners = [Subcampaign_Learner(budgets, l) for l in labels]
    # List to collect the reward at each time step
    rewards = []

    for t in range(T):
        # Sample from the Subcampaign_GP to get clicks estimations for each arm
        # and build the table to pass to Knapsack
        estimations = []
        for s in s_learners:
            estimate = [s.sample_from_GP(a) for a in budgets]
            # in this way the estimation of the budget equal to zero is always zero
            estimate[0] = 0
            # print(estimate)
            if(sum(estimate) == 0):
                estimate = [i * 1e-3 for i in range(n_arms)]
            estimations.append(estimate)
        # Knapsack return the super_arm as [(subcampaign, best_budget to assign), ..]
        super_arm = knapsack_optimizer(estimations)

        # print(super_arm)

        # For each subcampaign and budget related to it, use the budget to sample from the environment (click function)
        # Then use the sample obtained to update the current Subcampaing_GP
        super_arm_reward = 0
        for (subcampaign_id, budget_arm) in super_arm:
            reward = env.get_subcampaign_by_idx(
                subcampaign_id).aggr_sample(budget_arm)
            # print(reward)
            super_arm_reward += reward  # sum each arm reward in the reward of the super arm
            s_learners[subcampaign_id].update(budget_arm, reward)
        # this list contains the total reward for each time step
        rewards.append(super_arm_reward)

    gpts_rewards_per_experiment.append(rewards)
    # print(gpts_rewards_per_experiment)
    print(e+1)

plt.figure()
plt.ylabel("Reward - Regret")
plt.xlabel("t")
mean_exp = np.mean(gpts_rewards_per_experiment, axis=0)
opt = [opt_rewards_per_experiment for t in range(T)]
regret = opt-mean_exp
plt.plot(regret, 'r')
plt.plot(mean_exp, 'g')
plt.plot(opt, 'b')
plt.show()
