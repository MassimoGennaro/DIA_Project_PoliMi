import numpy as np
import matplotlib.pyplot as plt
import json

from Pricing.learner import *
from Pricing.modules import *
from Pricing.Pricing_Config_Manager import *
#from .utils_functions import *

class Experiment_4_5:
    def __init__(self, id):

        pri_env = Pricing_Config_Manager(id)
        self.categories = pri_env.get_indexed_categories()
        self.features = pri_env.features
        self.features_space = pri_env.feature_space
        self.p_categories = np.array(pri_env.probabilities)
        self.arms_candidates = np.array(pri_env.prices)
        self.n_arms = len(self.arms_candidates)
        
        self.experiments_logs = []
        self.beta_parameters_list = []
        
        self.ran = False
        
    def run_clairvoyant_context(self):
        # per ogni categoria calcolo il valore atteso di ogni arm
        exp_values_categories = np.multiply(self.p_categories, self.arms_candidates)
        # per ogni categoria trovo il valore atteso del best arm
        best_exp_value_categories = np.max(exp_values_categories, axis=1).T
        # Scelgo best arm per category
        opt_arm_categories = np.argmax(exp_values_categories, axis=1).T
        

        return best_exp_value_categories
    
    def run_clairvoyant_nocontext(self):
        # per ogni categoria calcolo il valore atteso di ogni arm aggregando le tre classi
        exp_values_categories = np.mean(np.multiply(self.p_categories, self.arms_candidates),axis=0)
        
        best_exp_value = np.max(exp_values_categories)
        
        return best_exp_value
    
    def run_experiment(self, n_experiments, horizon, week = -1):
        '''
        default week = -1
        if week > 0 performs context generation
        '''
        self.week = week
        for e in range(n_experiments):
            print("Perforiming experiment: {}".format(e+1))
            # ad ogni experiment creiamo un nuovo Environment
            
            # utilizziamo un environment adatto, rende la reward in base a category e candidato
            environment = Personalized_Environment(self.arms_candidates, self.p_categories)
            # utilizziamo un person_manager per gestire la creazione di nuove persone
            p_manager = Person_Manager(self.categories, self.p_categories, self.features)
            # utilizziamo un context_manager per gestire la gestione dei contesti e learner
            c_manager = Context_Manager(self.n_arms, self.features_space, self.categories, self.arms_candidates, week)
            # general gestisce la logica
            general = General(p_manager, c_manager, environment)

            # general itera per t round, restituendo le statistiche finali dell'esperimento
            experiment_log = general.play_experiment(horizon)

            # memorizzo per ogni esperimento i beta parameters finali
            self.beta_parameters_list.append(general.context_manager.contexts_set[0].learner.beta_parameters)

            # appendo le statistiche finali
            self.experiments_logs.append(experiment_log)
        self.ran = True
    
    def plot_regret_nocontext(self):
        if not self.ran:
            return "Run the experiment before plotting"
        total_regret_list = []
        # per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
        # meno valore atteso dell'arm scelto per tale categoria
        n_experiments = len(self.experiments_logs)
        best_exp_value = self.run_clairvoyant_nocontext()
        
        for e in range(n_experiments):
            #print("experiment: {}".format(e))
            regret_list = []
            regret_t = 0
            for log_t in self.experiments_logs[e]:
                #category_t = reward_t[0]
                #arm_chosen_t = reward_t[1]
                reward = log_t[2]
                
                regret_t = best_exp_value - reward
                regret_list.append(regret_t)
        
            total_regret_list.append(regret_list)
            # faccio la media del regret_t di ogni experiment
        average_regret_list = np.cumsum(np.mean(total_regret_list, axis=0))
        # print("average_regret_list lenght: {}".format(len(average_regret_list)))
        # print("cumulative average_regret_list lenght: {}\n".format(average_regret_list))

        # Stampa Grafico della Regret
        plt.figure(0)
        plt.ylabel("Cumulative Regret in t")
        plt.xlabel("t")
        plt.plot(average_regret_list, 'r')
        plt.legend(["Regret TS"])
        plt.show()
        
    def plot_regret_context(self):
        if not self.ran:
            return "Run the experiment before plotting"
        total_regret_list = []
        # per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
        # meno valore atteso dell'arm scelto per tale categoria
        n_experiments = len(self.experiments_logs)
        best_exp_value_categories = self.run_clairvoyant_context()
        
        for e in range(n_experiments):
            #print("experiment: {}".format(e))
            regret_list = []
            regret_t = 0
            for log_t in self.experiments_logs[e]:
                category_t = log_t[0]
                reward = log_t[2]
                best_exp_value = best_exp_value_categories[category_t]
                
                regret_t = best_exp_value - reward
                regret_list.append(regret_t)
        
            total_regret_list.append(regret_list)
            # faccio la media del regret_t di ogni experiment
        average_regret_list = np.cumsum(np.mean(total_regret_list, axis=0))
        # print("average_regret_list lenght: {}".format(len(average_regret_list)))
        # print("cumulative average_regret_list lenght: {}\n".format(average_regret_list))

        # Stampa Grafico della Regret
        plt.figure(0)
        plt.ylabel("Cumulative Regret in t")
        plt.xlabel("t")
        plt.plot(average_regret_list, 'r')
        plt.legend(["Regret TS with Context Generation"])
        plt.show()

    def plot_reward(self):
        n_experiments = len(self.experiments_logs)
        rewards_exp= []
        for e in range(n_experiments):
            
            rewards = []
            for log_t in self.experiments_logs[e]:
                reward = log_t[2]
                rewards.append(reward)
            
            rewards_exp.append(rewards)
        avg_reward_exp = np.cumsum(np.mean(rewards_exp, axis=0))

        plt.figure(0)
        plt.ylabel("Cumulative Reward in t")
        plt.xlabel("t")
        plt.plot(avg_reward_exp, 'r')
        plt.scatter(len(avg_reward_exp),np.max(avg_reward_exp))
        plt.legend(["Reward"])
        plt.show()
        print(np.max(avg_reward_exp))