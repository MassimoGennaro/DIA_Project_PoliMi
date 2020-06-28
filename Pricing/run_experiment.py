import numpy as np
import matplotlib.pyplot as plt
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

from learner import *
from modules import *


# setto il seed di random per la riproducibilità
random.seed(1234)

###### DATI DEL PROBLEMA #####

###### Dati delle Persone

# considero le categorie sempre come numeri interi
categories = {0: ("y", "f"), 1: ("a", "f"), 2: ("y", "u")}

# le features servono sono per capire a cosa corrispondono i valori
features = {"Age": ("y", "a"), "Familiarity": ("f", "u")}


# ogni tupla è in ordine secondo le variabili: (e.g. (prima var, seconda var, ecc))
feature_space = [("y", "f"),("a", "f"), ("y", "u") , ("a", "u")]

# ogni categoria ha una probabilità a priori
# sono equiprobabili
prior_categories = [1, 1, 1]

# l'environment è suddiviso in fasi, in cui in ciascuna cambiano le probabilità


# MATRICE PROBABILITà CLASSE - CANDIDATO
# modificando le matrici ottengo cambio dei migliori valori attesi
# BEST ARM è (1, 1, 1)
p_categories = np.array([[0.85, 0.85, 0.50, 0.25, 0.15],
                          [0.95, 0.90, 0.60, 0.05, 0.00],
                          [0.90, 0.85, 0.20, 0.15, 0.05]])

# BEST ARM è (0, 1, 2)
#p_categories = np.array([[0.999, 0.35, 0.25, 0.15, 0.10],
#                          [0.95, 0.90, 0.60, 0.05, 0.00],
#                          [0.90, 0.85, 0.60, 0.15, 0.05]])

###### Dati dei Candidati ######
# abbiamo 5 candidati di prezzzi diversi
n_arms = 5
# valori dei candidati
arms_candidates = np.array([5, 10, 15, 20, 25])

# per ogni categoria calcolo il valore atteso di ogni arm
exp_values_categories = np.multiply(p_categories, arms_candidates)

# per ogni categoria trovo il valore atteso del best arm
best_exp_value_categories = np.max(exp_values_categories, axis=1).T

# Scelgo best arm per category
opt_arm_categories = np.argmax(exp_values_categories, axis=1).T

### EXPERIMENT PARAMETERS ###

# T è il numero di persone
T = 5000

# week è ogni quante persone effettua split
#week = -1
#week = 6000
#week = 3000
#week = 1500
#week = 1000
#week = 500
#week = 300
#week = 150
#week = 100
#week = 50
#week = 25
week = 15

# numero esperimenti paralleli
n_experiments = 1

# STAMPO INFORMAZIONI UTILI PER ANALISI ESPERIMENTO

print("Informazioni\n")
# rpobabilità
print("A = \n", p_categories)
# candidati
print("B = \n", arms_candidates)
# valori attesi di ogni classe
print("A*B = \n", exp_values_categories)
# best valori attesi di ogni classe
print("max(A*B, axis=1) = \n", best_exp_value_categories)
# best candidati per ogni classe
print("argmax(A*B, axis=1) = \n", opt_arm_categories)

print("Horizon = {}, n_experiments = {}\n".format(T, n_experiments))

###### EFFETTUO ESPERIMENTI E CALCOLO MEDIA DELLA REGRET ######

# per ogni experiment, registro i valori attesi stimati da ogni contesto.
# quando calcolo la regret al tempo t, controllo la categoria e l'arm scelto
# da questi, ricavo la stima del valore atteso del contesto corrispondente
# se cat = 0 e arm = 1, controllo il valore atteso dell'arm 1 del contesto che si occupa di cat = 0

# registro la lista delle regret di ogni esperimento in questa lista
experiments_logs = []
# stimo i parametri beta di ogni arm
beta_parameters_experiments = {}

# CICLO ESPERIMENTI
for e in range(n_experiments):
    print("\nexperiment: {}".format(e))
    # creo environment
    #environment = Personalized_Environment(arms_candidates, p_categories, T)
    environment = Personalized_Environment(arms_candidates, p_categories)

    # person_manager gestisce creazione persone
    p_manager = Person_Manager(categories, p_categories, features)
    # context_manager gestisce contesti
    c_manager = Context_Manager(n_arms, feature_space, arms_candidates, week)
    # general gestisce tutto
    general = General(p_manager, c_manager, environment)

    # Itera per T round, restituendo le statistiche finali dell'esperimento
    exp_log = general.play_experiment(T)
    # appendo le statistiche finali
    experiments_logs.append(exp_log)

    # Registro beta parameters di tutti i contesti dell'esperimento
    contexts_experiment = [c for c in general.context_manager.contexts_set.values()]
    for c in contexts_experiment:
        print("contesto {}\nsubspace {}\nbeta_params\n{}\nrewards_log {}\n".format(c.context_id, c.subspace, c.learner.beta_parameters, len(c.rewards_log)))
    beta_parameters_experiments[e] = contexts_experiment

    


###### ANALISI DELLA REGRET ######

total_regret_list = []
# per ogni experiment trovo la lista dei regret ad ogni istante t
for e in range(n_experiments):
    regret_list = []
    regret_t = 0
    # trovo la lista dei regret di e
    for reward_t in experiments_logs[e]:
        # data la reward_t, trovo categoria persona e chosen arm
        category_t = reward_t[0]
        arm_chosen_t = reward_t[1]
        # dalla categoria ricavo quale fosse il best expected value e il chosen expected value
        best_exp_value = best_exp_value_categories[category_t]
        arm_exp_value = exp_values_categories[category_t][arm_chosen_t]
        # la regret è la differenza fra questi due expected reward
        regret_t = best_exp_value - arm_exp_value
        regret_list.append(regret_t)
    else:
        total_regret_list.append(regret_list)
else:
    # effettuo la media dei regret di ogni istante t per ciascuno degli esperimenti.
    average_regret_list = np.cumsum(np.mean(total_regret_list, axis=0))

# STAMPA REGRET MEDIA
plt.figure(0)
plt.title("Cumulative Regret with week = {}".format(week))
plt.ylabel("Cumulative Regret ")
plt.xlabel("time t")
plt.plot(average_regret_list, 'r')
if week == -1:
    plt.legend(["Regret TS"])
else:
    plt.legend(["Regret TS with Context Generation"])
#plt.show()
plt.savefig(dir_path + "\\Regret Medio week = {}".format(week))

#  QUI SI POSSONO OSSERVARE GLI ARMS PIù SCELTI IN TUTTI GLI EXPERIMENT
# quali sono gli arm più scelti in tutti gli experiment?

for i, e in beta_parameters_experiments.items():
    print("exp {}".format(i))


print("CORRECTLY ENDED")