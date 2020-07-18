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
        
    def run_clairvoyant(self):
        # per ogni categoria calcolo il valore atteso di ogni arm
        exp_values_categories = np.multiply(self.p_categories, self.arms_candidates)
        # per ogni categoria trovo il valore atteso del best arm
        best_exp_value_categories = np.max(exp_values_categories, axis=1).T
        # Scelgo best arm per category
        opt_arm_categories = np.argmax(exp_values_categories, axis=1).T
        

        return best_exp_value_categories
    
    def run_clairvoyant_aggregated(self):
        # per ogni categoria calcolo il valore atteso di ogni arm aggregando le tre classi
        exp_values_categories = np.mean(np.multiply(self.p_categories, self.arms_candidates),axis=0)
        
        best_exp_value = np.max(exp_values_categories)
        
        return best_exp_value
    
    def run_experiment(self, n_experiments, horizon, week = -1, beta_graph=False):
        '''
        default week = -1
        if week > 0 performs context generation
        '''
        # ogni experiment corrisponde a n esperimenti paralleli, con T persone.
        # per il grafico della regret, faccio la regret cumulativa di ognuno
        # e calcolo la media su n experiment.
        # ma experiments_logs ad ogni run deve prima essere svuotato, poi riusato.
        self.experiments_logs = []
        self.week = week
        self.beta_graph = beta_graph
        for e in range(n_experiments):
            print("Performing experiment: {}".format(e+1))
            # ad ogni experiment creiamo un nuovo Environment
            
            # utilizziamo un environment adatto, rende la reward in base a category e candidato
            environment = Personalized_Environment(self.arms_candidates, self.p_categories)
            # utilizziamo un person_manager per gestire la creazione di nuove persone
            p_manager = Person_Manager(self.categories, self.p_categories, self.features)
            # utilizziamo un context_manager per gestire la gestione dei contesti e learner
            c_manager = Context_Manager(self.n_arms, self.features_space, self.categories, self.arms_candidates, week, contexts_known = False)
            # general gestisce la logica
            general = General(p_manager, c_manager, environment)

            # general itera per t round, restituendo le statistiche finali dell'esperimento
            experiment_log = general.play_experiment(horizon)
            # memorizzo per ogni esperimento i beta parameters finali
            self.beta_parameters_list.append(general.context_manager.contexts_set[0].learner.beta_parameters)

            # appendo le statistiche finali
            self.experiments_logs.append(experiment_log)
        
        if(beta_graph):
            x_ticks = tuple(self.arms_candidates)
            x_1 = self.arms_candidates
            beta_exp = self.beta_parameters_list[0]
            opt_exp_values = np.mean(np.multiply(self.p_categories, self.arms_candidates),axis=0)
            print(opt_exp_values)
            y_1 = []
            ci = []
            for index,(alfa,beta) in enumerate(beta_exp):
                arm = self.arms_candidates[index]
                exp_value = alfa/(alfa+beta)
                y_1.append(exp_value*arm)
                variance = (alfa*beta)/((alfa+beta)**2*(alfa+beta+1))
                ci.append((arm)*(1.96*variance))
            
            plt.figure(0)
            plt.ylabel("Expected Reward (price * conversion_rate)")
            plt.xlabel("Prices")
            plt.errorbar(x_1,y_1,ci,capsize=7,linestyle="None", marker="o", markersize=1, mfc="blue", mec="blue")
            plt.scatter(x_1,opt_exp_values,marker="x",c="r")
            plt.show()
            
        
        #print("len(self.experiments_logs) =", len(self.experiments_logs))
        self.ran = True
    
    def plot_regret_aggregated(self):
        if not self.ran:
            return "Run the experiment before plotting"
        total_regret_list = []
        # per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
        # meno valore atteso dell'arm scelto per tale categoria
        exp_values_categories = np.mean(np.multiply(self.p_categories, self.arms_candidates),axis=0)
        n_experiments = len(self.experiments_logs)
        best_exp_value = self.run_clairvoyant_aggregated()
        
        for e in range(n_experiments):
            #print("experiment: {}".format(e))
            regret_list = []
            regret_t = 0
            for log_t in self.experiments_logs[e]:
                #category_t = log_t[0]
                arm_chosen_t = log_t[1]
                reward = log_t[2]
                val_atteso_stimato_arm = log_t[3]
                val_atteso_arm = exp_values_categories[arm_chosen_t]
                
                regret_t = best_exp_value - val_atteso_arm
                #regret_t = best_exp_value - val_atteso_stimato_arm
                #regret_t = best_exp_value - reward
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
        plt.legend(["Regret TS aggregated clairvoyant"])
        plt.show()
        
    def plot_regret(self, comparison = False):
        if not self.ran:
            return "Run the experiment before plotting"
        total_regret_list = []
        # per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
        # meno valore atteso dell'arm scelto per tale categoria
        n_experiments = len(self.experiments_logs)

        # matrice valori attesi categoria-arm
        exp_values_categories = np.multiply(self.p_categories, self.arms_candidates)


        best_exp_value_categories = self.run_clairvoyant()
        
        for e in range(n_experiments):
            #print("experiment: {}".format(e))
            regret_list = []
            regret_t = 0
            for log_t in self.experiments_logs[e]:
                category_t = log_t[0]
                pulled_arm = log_t[1]
                reward = log_t[2]
                val_atteso_stimato_arm = log_t[3]
                best_exp_value = best_exp_value_categories[category_t]
                val_atteso_arm = exp_values_categories[category_t][pulled_arm]
                
                #regret_t = best_exp_value - reward
                #regret_t = best_exp_value - val_atteso_stimato_arm
                regret_t = best_exp_value - val_atteso_arm
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
        plt.annotate(np.max(avg_reward_exp),(len(avg_reward_exp),np.max(avg_reward_exp)),arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"), xytext=(len(avg_reward_exp)-1000,np.max(avg_reward_exp)-10000))
        plt.legend(["Reward"])
        plt.show()