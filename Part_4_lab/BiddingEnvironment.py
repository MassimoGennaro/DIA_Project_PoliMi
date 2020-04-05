import numpy as np

def bid_to_clicks(x):
    return 100*(1.0 - np.exp(-4*x + 3*x**3))

class BiddingEnv():
    def __init__(self, bids, sigma):
        self.bids = bids
        self.means = bid_to_clicks(bids)
        self.sigmas = np.ones(len(bids))*sigma
        
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm],self.sigmas[pulled_arm])
    