# Learners

import numpy as np
import math
from matplotlib import pyplot

class Learner:
    def __init__(self,n_arms):
        self.n_arms = n_arms
        self.t = 0
        #self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])


    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards,reward)

########################################


class TS_Learner(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms,2))



    def pull_arm(self):
        # scelgo braccio con valore atteso più alto
        idx =  np.argmax(np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1]))
        return idx

    def prob_succ_arm(self, arm):
        arm_successes = self.beta_parameters[arm][0]
        arm_failures = self.beta_parameters[arm][1]
        
        return arm_successes/(arm_successes + arm_failures)

    def expected_value(self, arm, candidate_value):
    	#calcolo la sua probabilità di successo
        arm_successes = self.beta_parameters[arm][0]
        arm_failures = self.beta_parameters[arm][1]
        arm_prob_success = self.prob_succ_arm(arm)
        # calcolo il suo valore atteso, deve essere usato per la split condition
        expected_value = arm_prob_success * candidate_value
        return expected_value

        #Dati i valori dei candidati (prezzi) restituisce il best arm che è dato dal prodotto value*prob_success

    # best arm è indice dell'arm con il miglir valore atteso stimato
    def best_arm(self, candidates_values):
        best = 0
        best_value = self.expected_value(0,candidates_values[0]) #this should crash whenever the inputs have size zero
        for i in range(self.n_arms):
            if self.expected_value(i, candidates_values[i]) > best_value:
                best = i
                bast_value = self.expected_value(i, candidates_values[i])
        return best

    def best_exp_value(self, candidates_values):
        best_arm = self.best_arm(candidates_values)
        return self.expected_value(best_arm, candidates_values[best_arm])

    def best_arm_lower_bound(self, candidates_values):
        best_arm = self.best_arm(candidates_values)
        exp_value = self.expected_value(best_arm, candidates_values[best_arm])
        succ_arm = self.prob_succ_arm(best_arm)
        alfa_best_arm = self.beta_parameters[best_arm][0]
        beta_best_arm = self.beta_parameters[best_arm][1]
        # il lower bound deve decrescere nel tempo secondo t...
        # infatti più passa il tempo e più dovremmo essere sicuri delle nostre stime
        # altrimenti ad ogni context generetaion perdiamo sicurezza nelle stime, ma non è così
        minus = -pow(-math.log(succ_arm * (1-succ_arm)) / (2 * (alfa_best_arm + beta_best_arm)), 0.5) 
        # TODO: sostituisci (alfa_best_arm + beta_best_arm) con il self.t
        # NEW!
        #minus = -pow(-math.log(succ_arm * (1-succ_arm)) / (2 * (self.t)), 0.5) 
        return exp_value - minus
########################################


class TS_Learner_candidate(TS_Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        # parto per ogni arm con avere valore atteso uguale a 0.
        # tale valore va aggiornato con ogni update
        self.exp_value_arm = {i:0 for i in range(n_arms)}


    # devo scegliere l'arm in base al valore atteso fino a questo turno!
    def pull_arm(self, candidates_values):
        # effettuo samples delle beta normale
        beta_samples = np.random.beta(self.beta_parameters[:,0],self.beta_parameters[:,1])

        # calcolo expected reward con valore atteso del candidato
        expected_reward = np.multiply(beta_samples, candidates_values)

        # scelgo braccio con expected reward più alto
        idx =  np.argmax(expected_reward)

        return idx
    
    def update(self, pulled_arm, reward):
        self.t += 1
        esito = 0
        # se reward è positiva, è un successo. Altrimento è insuccesso
        if reward > 0:
            esito = 1
        elif reward <= 0:
            esito = 0

        self.update_observations(pulled_arm,reward)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm,0] + esito
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm,1] + 1.0 - esito

    # aggiorno la funzione per considerare la media di ogni arm.
    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)
