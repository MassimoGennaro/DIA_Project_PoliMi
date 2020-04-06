from BudgetEnvironment import *
import numpy as np
from GPTS_Learner import *

n_arms = 5
min_budget = 1
max_budget = 5

T = 60
gpts_rewards_per_experiment = []

budgets = np.linspace(min_budget, max_budget, n_arms)

env = BudgetEnvironment(budgets)
gpts_learner = GPTS_Learner(n_arms, budgets)

for t in range(T):
    pulled_arm = gpts_learner.pull_arm()
    reward = env.round(pulled_arm,'FaY')
    gpts_learner.update(pulled_arm, reward)
