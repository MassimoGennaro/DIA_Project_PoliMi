from environment.BudgetEnvironment import *
import numpy as np
from learners.NS_Subcampaign_Learner import NS_Subcampaign_Learner
from learners.Subcampaign_Learner import Subcampaign_Learner
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
phases = ['M', 'E', 'W']
phases_ts = []
T = 56

phase_1_ts = [t for t in range(T-16) if (t % 8) < 4]
phases_ts.append(phase_1_ts)
phase_2_ts = [t for t in range(T-16) if (t % 8) >= 4]
phases_ts.append(phase_2_ts)
phase_3_ts = [t for t in range(T-16, T)]
phases_ts.append(phase_3_ts)

phase_dict = {t: 'M' for t in phase_1_ts}
phase_dict.update({t: 'E' for t in phase_2_ts})
phase_dict.update({t: 'W' for t in phase_3_ts})


n_experiments = 10


# Optimal Solution
opt_rewards_per_experiment = []
env_opt = BudgetEnvironment(subcampaigns)
for t in range(T):
    optimal_super_arm_reward = 0
    real_values = []
    for subc in env_opt.subcampaigns:
        real_values.append([subc.real_function_disaggr(
            b, phase_dict[t]) for b in budgets])
        optimal_super_arm = knapsack_optimizer(real_values)

    for (subcampaign_id, budget_arm) in optimal_super_arm:
        optimal_reward = env_opt.get_subcampaign_by_idx(
            subcampaign_id).real_function_disaggr(budget_arm, phase_dict[t])
        optimal_super_arm_reward += optimal_reward

    opt_rewards_per_experiment.append(optimal_super_arm_reward)

# print(opt_rewards_per_experiment)


# Contains the rewards for each experiment (each element is a list of T rewards)
SWgpts_rewards_per_experiment = []
gpts_rewards_per_experiment = []

for e in range(n_experiments):
    # Create the BudgetEnvironment usint the list of sucampaigns
    env = BudgetEnvironment(subcampaigns)

    # Create a list of Subcampaign_GP
    # Sliding Window
    SW_s_learners = [NS_Subcampaign_Learner(budgets, l, T) for l in labels]
    # Non Sliding Windows
    s_learners = [Subcampaign_Learner(budgets, l) for l in labels]
    # List to collect the reward at each time step
    sw_rewards = []
    rewards = []
    for t in range(T):
        
        ### SLIDING WINDOW ###
        
        # Sample from the Subcampaign_GP to get clicks estimations for each arm
        # and build the table to pass to Knapsack
        estimations = []
        for s in SW_s_learners:
            estimate = [s.sample_from_GP(a) for a in budgets]
            # in this way the estimation of the budget equal to zero is always zero
            estimate[0] = 0
            
            if(sum(estimate) == 0):
                estimate = [i * 1e-3 for i in range(n_arms)]
            estimations.append(estimate)
        # Knapsack return the super_arm as [(subcampaign, best_budget to assign), ..]
        super_arm = knapsack_optimizer(estimations)

        

        # For each subcampaign and budget related to it, use the budget to sample from the environment (click function)
        # Then use the sample obtained to update the current Subcampaing_GP
        super_arm_reward = 0
        for (subcampaign_id, budget_arm) in super_arm:
            reward = env.get_subcampaign_by_idx(
                subcampaign_id).disaggr_sample(budget_arm, phase_dict[t])
            
            super_arm_reward += reward  # sum each arm reward in the reward of the super arm
            SW_s_learners[subcampaign_id].update(budget_arm, reward, t)
        # this list contains the total reward for each time step
        sw_rewards.append(super_arm_reward)
        
        
        ### NON SLIDING WINDOW ###
        

    
        # Sample from the Subcampaign_GP to get clicks estimations for each arm
        # and build the table to pass to Knapsack
        estimations = []
        for s in s_learners:
            estimate = [s.sample_from_GP(a) for a in budgets]
            # in this way the estimation of the budget equal to zero is always zero
            estimate[0] = 0
            
            if(sum(estimate) == 0):
                estimate = [i * 1e-3 for i in range(n_arms)]
            estimations.append(estimate)
        # Knapsack return the super_arm as [(subcampaign, best_budget to assign), ..]
        super_arm = knapsack_optimizer(estimations)

        

        # For each subcampaign and budget related to it, use the budget to sample from the environment (click function)
        # Then use the sample obtained to update the current Subcampaing_GP
        super_arm_reward = 0
        for (subcampaign_id, budget_arm) in super_arm:
            reward = env.get_subcampaign_by_idx(
                subcampaign_id).disaggr_sample(budget_arm,phase_dict[t])
            
            super_arm_reward += reward  # sum each arm reward in the reward of the super arm
            s_learners[subcampaign_id].update(budget_arm, reward)
        # this list contains the total reward for each time step
        rewards.append(super_arm_reward)

    
    SWgpts_rewards_per_experiment.append(sw_rewards)
    gpts_rewards_per_experiment.append(rewards)

    
    
    print(e+1)

plt.figure()
plt.ylabel("Number of Clicks")
plt.xlabel("t")

mean_exp_SW = np.mean(SWgpts_rewards_per_experiment, axis=0)
mean_exp = np.mean(gpts_rewards_per_experiment, axis = 0)
opt = opt_rewards_per_experiment
regret_SW = opt-mean_exp_SW
regret = opt-mean_exp
# plt.plot(np.cumsum(regret), 'r' , label='Cumulative Regret')
# plt.plot(np.cumsum(mean_exp), 'b',label='Cumulative Expected Rewards')
# plt.plot(np.cumsum(opt), 'g',label='Cumulative Optimal Reward')

plt.plot(regret_SW, 'r', label='Regret SW')
plt.plot(regret, 'r--', label = 'Regret no SW')
plt.plot(mean_exp_SW, 'b', label='Expected Reward SW')
plt.plot(mean_exp, 'b--', label='Expected Reward no SW')
plt.plot(opt, 'g', label='Optimal Reward')

plt.legend(loc="upper left")
plt.show()
