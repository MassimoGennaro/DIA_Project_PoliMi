from environment.CampaignEnvironment import *
from learners.NS_Subcampaign_Learner import NS_Subcampaign_Learner
from learners.Subcampaign_Learner import Subcampaign_Learner
from knapsack.knapsack import *
import numpy as np
import matplotlib.pyplot as plt

max_budget = 5.0
n_arms = int(max_budget + 1)
budgets = np.linspace(0.0, max_budget, n_arms)
sigma = 2.0

feature_labels = ["Young-Familiar", "Adult-Familiar", "Young-NotFamiliar"]

# Phases
phase_labels = ["Morning", "Evening", "Weekend"]
phase_dict = ([phase_labels.index("Morning")]*4 + [phase_labels.index("Evening")]*4)*5 + [phase_labels.index("Weekend")]*16
phase_weights = [5/14, 5/14, 4/14]


T = 56

n_experiments = 2

########################
# Clairvoyant Solution #
########################

opt_env = Campaign(budgets, phases=phase_labels, weights=phase_weights, sigma=0.0)
for feature_label in feature_labels:
    opt_env.add_subcampaign(label=feature_label)


optimal_super_arm_reward_phase = []
for phase in range(len(phase_labels)):
    real_values = opt_env.round_all(phase=phase)
    optimal_super_arm = knapsack_optimizer(real_values)

    optimal_super_arm_reward = 0
    for (subc_id, pulled_arm) in enumerate(optimal_super_arm):
        optimal_reward = opt_env.subcampaigns[subc_id].round(pulled_arm, phase=phase)
        optimal_super_arm_reward += optimal_reward

    optimal_super_arm_reward_phase.append(optimal_super_arm_reward)

opt_rewards_per_experiment = []
for t in range(T):
    opt_rewards_per_experiment.append(optimal_super_arm_reward_phase[phase_dict[t]])

#########################
# Experimental Solution #
#########################

# Contains the rewards for each experiment (each element is a list of T rewards)
SWgpts_rewards_per_experiment = []
gpts_rewards_per_experiment = []

for e in range(n_experiments):
    print("experiment: ", str(e+1))

    # Create the BudgetEnvironment usint the list of sucampaigns
    env = Campaign(budgets, phases=phase_labels, weights=phase_weights, sigma=sigma)

    # list of GP-learners
    subc_learners = []
    SW_s_learners = []

    # add subcampaings to the environment
    # and create a GP-learner for each subcampaign
    for feature_label in feature_labels:
        env.add_subcampaign(label=feature_label)
        # Non Sliding Windows
        subc_learners.append(Subcampaign_Learner(arms=budgets, label=feature_label))
        # Sliding Window
        SW_s_learners.append(NS_Subcampaign_Learner(arms=budgets, label=feature_label, horizon=T))

    # List to collect the reward at each time step
    sw_rewards = []
    rewards = []
    for t in range(T):

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
            arm_reward = env.subcampaigns[subc_id].round(pulled_arm, phase=phase_dict[t])
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
            arm_reward = env.subcampaigns[subc_id].round(pulled_arm, phase=phase_dict[t])
            super_arm_reward += arm_reward
            subc_learners[subc_id].update(pulled_arm, arm_reward)

        # store the reward for this timestamp
        rewards.append(super_arm_reward)

    SWgpts_rewards_per_experiment.append(sw_rewards)
    gpts_rewards_per_experiment.append(rewards)


plt.figure()
plt.ylabel("Number of Clicks")
plt.xlabel("t")

mean_exp_SW = np.mean(SWgpts_rewards_per_experiment, axis=0)
mean_exp = np.mean(gpts_rewards_per_experiment, axis=0)
opt = opt_rewards_per_experiment
regret_SW = opt - mean_exp_SW
regret = opt - mean_exp
# plt.plot(np.cumsum(regret), 'r' , label='Cumulative Regret')
# plt.plot(np.cumsum(mean_exp), 'b',label='Cumulative Expected Rewards')
# plt.plot(np.cumsum(opt), 'g',label='Cumulative Optimal Reward')

plt.plot(regret_SW, 'r', label='Regret SW')
plt.plot(regret, 'r--', label='Regret no SW')
plt.plot(mean_exp_SW, 'b', label='Expected Reward SW')
plt.plot(mean_exp, 'b--', label='Expected Reward no SW')
plt.plot(opt, 'g', label='Optimal Reward')

plt.legend(loc="upper left")
plt.show()
