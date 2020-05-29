import numpy as np
import matplotlib.pyplot as plt
import json

from Pricing.learner import *
from Pricing.modules import *
#from .utils_functions import *

class Experiment_4_5:
    def __init__(self, arms_candidates, id):
        
        with open('Pricing/configs/pricing_env.json') as json_file:
            data = json.load(json_file)
        campaign = data["campaigns"][id]
        
        
        categories = {i:tuple(campaign["categories"][i]) for i in range(len(campaign["categories"]))}
        self.categories = categories
        self.features = campaign["features"]
        feature_space = [tuple(campaign["features_space"][i]) for i in range(len(campaign["features_space"]))]
        self.features_space = feature_space
        self.p_categories = np.array(campaign["p_categories"])
        self.arms_candidates = np.array(arms_candidates)
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
        return exp_values_categories, best_exp_value_categories
    
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
    
    def plot_regret(self):
        if not self.ran:
            return "Run the experiment before plotting"
        total_regret_list = []
        # per ogni persona con una categoria, regret_t è valore atteso migliore della categoria
        # meno valore atteso dell'arm scelto per tale categoria
        n_experiments = len(self.experiments_logs)
        exp_values_categories, best_exp_value_categories = self.run_clairvoyant()
        
        for e in range(n_experiments):
            #print("experiment: {}".format(e))
            regret_list = []
            regret_t = 0
            for reward_t in self.experiments_logs[e]:
                category_t = reward_t[0]
                arm_chosen_t = reward_t[1]
                best_exp_value = best_exp_value_categories[category_t]
                arm_exp_value = exp_values_categories[category_t][arm_chosen_t]
                regret_t = best_exp_value - arm_exp_value
                regret_list.append(regret_t)
            else:
                total_regret_list.append(regret_list)
        else:
            # faccio la media del regret_t di ogni experiment
            average_regret_list = np.cumsum(np.mean(total_regret_list, axis=0))
            # print("average_regret_list lenght: {}".format(len(average_regret_list)))
            # print("cumulative average_regret_list lenght: {}\n".format(average_regret_list))

        # Stampa Grafico della Regret
        plt.figure(0)
        plt.ylabel("Cumulative Regret in t")
        plt.xlabel("t")
        plt.plot(average_regret_list, 'r')
        if self.week == -1:
            plt.legend(["Regret TS"])
        else:
            plt.legend(["Regret TS with Context Generation"])
        plt.show()
        
        # TODO : add other statistics of the experiments