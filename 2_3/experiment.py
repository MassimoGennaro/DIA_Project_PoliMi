from environment.CampaignEnvironment import *
from learners.Subcampaign_Learner import *
from knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt


# Budget settings
max_budget = 5.0

n_arms = int(max_budget + 1)
budgets = np.linspace(0.0, max_budget, n_arms)


# Phase settings
phase_labels = ["Morning", "Evening", "Weekend"]
phase_weights = [5/14, 5/14, 4/14]  # the sum must be equal to 1
T = 20  # Time Horizon


# Class settings
feature_labels = ["Young-Familiar", "Adult-Familiar", "Young-NotFamiliar"]


# Experiment settings
n_experiments = 2   # number of experiments
sigma = 2.0         # sampling variance


########################
# Clairvoyant Solution #
########################

opt_env = Campaign(budgets, phases=phase_labels, weights=phase_weights, sigma=0.0)
for feature_label in feature_labels:
    opt_env.add_subcampaign(label=feature_label)
real_values = opt_env.round_all()
opt_super_arm = knapsack_optimizer(real_values)
opt_super_arm_reward = 0
for (subc_id, pulled_arm) in enumerate(opt_super_arm):
    reward = opt_env.subcampaigns[subc_id].round(pulled_arm)
    opt_super_arm_reward += reward


#########################
# Experimental Solution #
#########################

# Contains the rewards for each experiment (each element is a list of T rewards)
gpts_rewards_per_experiment = []

for e in range(0, n_experiments):
    print("experiment: ", str(e+1))

    # create the environment
    env = Campaign(budgets, phases=phase_labels, weights=phase_weights, sigma=sigma)

    # list of GP-learners
    subc_learners = []

    # add subcampaings to the environment
    # and create a GP-learner for each subcampaign
    for feature_label in feature_labels:
        env.add_subcampaign(label=feature_label)
        subc_learners.append(Subcampaign_Learner(arms=budgets, label=feature_label))

    # rewards for each time step
    rewards = []

    for t in range(0, T):
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

    gpts_rewards_per_experiment.append(rewards)

plt.figure()
plt.ylabel("Number of Clicks")
plt.xlabel("t")

opt = [opt_super_arm_reward]*T
mean_exp = np.mean(gpts_rewards_per_experiment, axis=0)
regret = opt_super_arm_reward-mean_exp

# plt.plot(np.cumsum(regret), 'r' , label='Cumulative Regret')
# plt.plot(np.cumsum(mean_exp), 'b',label='Cumulative Expected Rewards')
# plt.plot(np.cumsum(opt), 'g',label='Cumulative Optimal Reward')
plt.plot(opt, 'g', label='Optimal Reward')
plt.plot(mean_exp, 'b', label='Expected Reward')
plt.plot(regret, 'r', label='Regret')

plt.legend(loc="upper left")
plt.show()
