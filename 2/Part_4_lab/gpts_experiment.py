import numpy as np
import matplotlib.pyplot as plt
from BiddingEnvironment import BiddingEnv
from GTS_Learner import *
from GPTS_Learner import *

n_arms = 20
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 60
n_experiments = 10
gts_rewards_per_experiment = []
gpts_rewards_per_experiment = []

for e in range(n_experiments):
    env = BiddingEnv(bids, sigma)
    gts_learner = GTS_Learner(n_arms)
    gpts_learner = GPTS_Learner(n_arms, bids)
    for t in range(T):
        # Gaussian Thompson Sampling
        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gts_learner.update(pulled_arm, reward)
        
        # GP Thompson Sampling
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm,reward)
        
    gts_rewards_per_experiment.append(gts_learner.collected_rewards)
    gpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
    print('Finished {} experiment'.format(e+1) )

opt = np.max(env.means)
plt.figure()
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt-gts_rewards_per_experiment, axis = 0)), 'r')
plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis = 0)), 'g')
plt.legend(['GTS','GPTS'])
plt.show()